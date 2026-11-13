from __future__ import annotations

import math
import torch
from pruning.nnprune.kronecker import dot, masked_cg
from pruning.nnprune.canonicalize.gated import GatedGroup, gated_dead


def ordinary(value, penalized, penalty, curvature, neurons_only):
    """Coordinatewise prox and its cost; with neurons_only, single coefficients are kept and simply charged."""
    if neurons_only:
        return value, torch.where(penalized & value.ne(0), penalty, 0.)
    squared = .5 * curvature * value.square()
    return value.masked_fill(penalized & squared.le(penalty), 0.), torch.where(penalized, squared.clamp_max(penalty), 0.)


def canonical_nonzeros(values, penalized, pair=None):
    if isinstance(pair, GatedGroup):
        gate, up, down = (values[name] for name in pair.names)
        dead = gated_dead(gate, up, down)
        return (int((gate[~dead].ne(0) & penalized[pair.gate][~dead]).sum())
                + int((up[~dead].ne(0) & penalized[pair.up][~dead]).sum())
                + int((down[:, ~dead].ne(0) & penalized[pair.down][:, ~dead]).sum()))
    if pair is not None and isinstance(pair[0], tuple):
        live = {name: value.ne(0) for name, value in values.items()}
        while True:
            changed = False
            for left, right, fan_in in pair:
                dead = ~live[left][:, :fan_in].any(1) | ~live[right][:, :len(live[left])].any(0)
                changed |= bool(live[left][dead].any() | live[right][:, :len(live[left])][:, dead].any())
                live[left][dead] = False
                live[right][:, :len(live[left])][:, dead] = False
            if not changed:
                return sum(int((mask & penalized[name]).sum()) for name, mask in live.items())
    count = sum(int(value[penalized[name]].count_nonzero()) for name, value in values.items())
    if pair is not None:
        left, right, fan_in = pair
        incoming, outgoing = values[left], values[right][:, :len(values[left])]
        dead = incoming[:, :fan_in].eq(0).all(1) | outgoing.eq(0).all(0)
        count -= int(incoming[dead][penalized[left][dead]].count_nonzero())
        count -= int(outgoing[:, dead][penalized[right][:, :len(incoming)][:, dead]].count_nonzero())
    return count


def proximal_step(z, penalty, curvature, penalized, pair=None, fixed=None, neurons_only=False):
    if fixed is not None:
        z = {name: value.masked_fill(fixed[name], 0.) for name, value in z.items()}
    if pair is not None and isinstance(pair[0], tuple):
        return chain_proximal(z, penalty, curvature, penalized, pair, neurons_only)
    values, costs = {}, {}
    for name, value in z.items():
        values[name], costs[name] = ordinary(value, penalized[name], penalty, curvature, neurons_only)
    if isinstance(pair, GatedGroup):
        gate, up, down = (z[name] for name in pair.names)
        kept = costs[pair.gate].sum(1) + costs[pair.up].sum(1) + costs[pair.down].sum(0)
        choice = torch.stack((kept, .5 * curvature * gate.square().sum(1), .5 * curvature * up.square().sum(1),
                              .5 * curvature * down.square().sum(0))).argmin(0)
        dead = choice.ne(0)
        values[pair.gate][dead], values[pair.up][dead], values[pair.down][:, dead] = gate[dead], up[dead], down[:, dead]
        values[pair.gate][choice.eq(1)] = 0
        values[pair.up][choice.eq(2)] = 0
        values[pair.down][:, choice.eq(3)] = 0
    elif pair is not None:
        left, right, fan_in = pair
        incoming, outgoing = z[left], z[right][:, :len(z[left])]
        kept = costs[left].sum(1) + costs[right][:, :len(incoming)].sum(0)
        # A constant or disconnected neuron contributes zero retained coefficients after folding.
        choice = torch.stack((kept, .5 * curvature * incoming[:, :fan_in].square().sum(1),
                              .5 * curvature * outgoing.square().sum(0))).argmin(0)
        constant, disconnected = choice.eq(1), choice.eq(2)
        values[left][constant] = incoming[constant]
        values[left][constant, :fan_in] = 0.
        values[right][:, :len(incoming)][:, constant] = outgoing[:, constant]
        values[left][disconnected] = incoming[disconnected]
        values[right][:, :len(incoming)][:, disconnected] = 0.
    return values


def chain_proximal(z, penalty, curvature, penalized, pairs, neurons_only):
    states = [torch.zeros(len(z[left]), dtype=torch.long, device=z[left].device) for left, _, _ in pairs]
    plain = {name: ordinary(value, penalized[name], penalty, curvature, neurons_only)[0] for name, value in z.items()}

    def evaluate():
        forced = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in z.items()}
        exempt = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in z.items()}
        for (left, right, fan_in), state in zip(pairs, states):
            dead = state.ne(0)
            forced[left][:, :fan_in] |= state.eq(1)[:, None]
            forced[right][:, :len(state)] |= state.eq(2)[None, :]
            exempt[left] |= dead[:, None]
            exempt[right][:, :len(state)] |= dead[None, :]
        values, costs = {}, {}
        for name, value in z.items():
            kept, cost = ordinary(value, penalized[name] & ~exempt[name], penalty, curvature, neurons_only)
            costs[name] = torch.where(forced[name], .5 * curvature * value.square(), cost)
            values[name] = kept.masked_fill(forced[name], 0.)
        return values, costs

    # Shared middle matrices couple adjacent hidden layers. Layer-coordinate descent
    # keeps that coupling and counts each coefficient once; it is not an exact prox.
    for _ in range(32):
        changed = False
        for index, (left, right, _) in enumerate(pairs):
            previous = states[index].clone()
            choices = []
            for choice in range(3):
                states[index].fill_(choice)
                _, costs = evaluate()
                choices.append(costs[left].sum(1) + costs[right][:, :len(previous)].sum(0))
            states[index] = torch.stack(choices).argmin(0)
            changed |= not torch.equal(previous, states[index])
        if not changed:
            break
    candidate = evaluate()[0]

    def objective(values):
        return (.5 * curvature * sum(float((value - z[name]).square().sum()) for name, value in values.items())
                + penalty * canonical_nonzeros(values, penalized, pairs))
    return candidate if objective(candidate) <= objective(plain) else plain


@torch.no_grad()
def minimize_l0(reference, metric, penalty, penalized, units, options, *, floor, fixed=None,
                accurate=False, neurons_only=False):
    """Penalized proximal gradient over canonical units sharing one metric.

    The fast solver keeps its step from `floor` (a Lipschitz estimate) and ends with one capped refit; the accurate
    solver adapts the step from `floor` (the largest diagonal), retries it before accepting stationarity, and requires
    its support refits to converge."""
    def gradient(values):
        return metric.product({name: values[name] - reference[name] for name in values})

    def evaluate(values, gradients):
        loss = .5 * dot({name: values[name] - reference[name] for name in values}, gradients)
        return loss, loss + penalty * count(values)

    def count(values):
        return sum(canonical_nonzeros({name: values[name] for name in names}, {name: penalized[name] for name in names}, pair)
                   for names, pair in units)

    def prox(z, curvature):
        result = {}
        for names, pair in units:
            result.update(proximal_step({name: z[name] for name in names}, penalty, curvature,
                                        {name: penalized[name] for name in names}, pair,
                                        {name: fixed[name] for name in names} if fixed is not None else None, neurons_only))
        return result

    def refit(values):
        free = {name: (value.ne(0) | ~penalized[name]) & (~fixed[name] if fixed is not None else True)
                for name, value in values.items()}
        refined, record = masked_cg(metric, reference, values, free, rtol=options.refit_rtol,
                                    iterations=options.refit_iterations, strict=accurate)
        refined_gradient = gradient(refined)
        return refined, refined_gradient, *evaluate(refined, refined_gradient), record

    value = {name: weight.clone() for name, weight in reference.items()}
    gradients = {name: torch.zeros_like(weight) for name, weight in reference.items()}
    loss, objective = 0., penalty * count(value)
    curvature, backtracks, converged, polished, restart, refits = floor, 0, False, False, False, []
    epsilon = 16 * torch.finfo(next(iter(value.values())).dtype).eps
    for iteration in range(1, options.max_iterations + 1):
        if accurate:
            curvature = floor if restart else max(floor, curvature / 2)
        restart, started_at_floor = False, curvature == floor
        for _ in range(options.max_backtracks):
            candidate = prox({name: weight - gradients[name] / curvature for name, weight in value.items()}, curvature)
            candidate_gradient = gradient(candidate)
            step = {name: weight - value[name] for name, weight in candidate.items()}
            remainder = dot(step, {name: candidate_gradient[name] - gradients[name] for name in step})
            bound = curvature * dot(step, step)
            candidate_loss, candidate_objective = evaluate(candidate, candidate_gradient)
            if (math.isfinite(candidate_objective) and candidate_objective <= objective and
                    remainder <= bound + epsilon * (abs(remainder) + abs(bound))):
                break
            curvature *= 2
            backtracks += 1
        else:
            raise RuntimeError("L0 proximal backtracking could not find a descent step")
        same_support = all(torch.equal(candidate[name].eq(0), weight.eq(0)) for name, weight in value.items())
        small_step = dot(step, step) <= options.rtol ** 2 * max(dot(value, value), torch.finfo(torch.float64).tiny)
        value, gradients, loss, objective = candidate, candidate_gradient, candidate_loss, candidate_objective
        polished &= same_support
        if same_support and small_step:
            if accurate and options.refit_iterations and not polished:
                refined, refined_gradient, refined_loss, refined_objective, record = refit(value)
                refits.append(record)
                if math.isfinite(refined_objective) and refined_objective <= objective:
                    value, gradients, loss, objective = refined, refined_gradient, refined_loss, refined_objective
                polished = restart = True
                continue
            if accurate and not started_at_floor:
                restart = True
                continue
            converged = True
            break
    refit_accepted = False
    if options.refit_iterations:
        refined, _, refined_loss, refined_objective, record = refit(value)
        refits.append(record)
        if math.isfinite(refined_objective) and refined_objective <= objective:
            value, loss, objective, refit_accepted = refined, refined_loss, refined_objective, True
    return value, {"iterations": iteration, "proximal_converged": converged, "backtracks": backtracks,
                   "lipschitz": curvature, "predicted_fisher_damage": loss, "penalized_objective": objective,
                   "nonzeros": count(value), "penalty": penalty, "refits": refits, "refit_accepted": refit_accepted}
