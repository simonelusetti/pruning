from __future__ import annotations

import math
import torch
from pruning.nnprune.factorized_fisher import complement_update
from pruning.nnprune.canonicalize.gated import GatedGroup, gated_dead, gated_proximal


def initial_curvature(a, b):
    return max(float(a.diag().max() * b.diag().max()), torch.finfo(a.dtype).tiny)


def dot(first, second):
    return float((first * second).sum(dtype=torch.float64))


def support_refit(value, reference, a, b, penalized, iterations, rtol, fixed=None, inverses=None):
    free = value.ne(0) | ~penalized
    if fixed is not None:
        free &= ~fixed
    delta, record = complement_update(
        reference, a, b, ~free, rtol=rtol, max_iterations=iterations,
        initial_update=value-reference,
        a_inverse=inverses[0] if inverses is not None else None,
        b_inverse=inverses[1] if inverses is not None else None)
    return (reference + delta).masked_fill_(~free, 0.), record["iterations"], record["relative_residual"]


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


def proximal_step(z, penalty, curvature, penalized, pair=None, fixed=None):
    if fixed is not None:
        z = {name: value.masked_fill(fixed[name], 0.) for name, value in z.items()}
    if isinstance(pair, GatedGroup):
        return gated_proximal(z, penalty, curvature, penalized, pair)
    if pair is not None and isinstance(pair[0], tuple):
        return chain_proximal(z, penalty, curvature, penalized, pair)
    values = {name: value.masked_fill(penalized[name] & value.square().le(2 * penalty / curvature), 0.)
              for name, value in z.items()}
    if pair is not None:
        left, right, fan_in = pair
        incoming, outgoing = z[left], z[right][:, :len(z[left])]
        costs = {name: torch.where(penalized[name], (.5 * curvature * value.square()).clamp_max(penalty), 0.)
                 for name, value in z.items()}
        ordinary = costs[left].sum(1) + costs[right][:, :len(incoming)].sum(0)
        # A constant or disconnected neuron contributes zero retained coefficients after folding.
        constant = .5 * curvature * incoming[:, :fan_in].square().sum(1)
        disconnected = .5 * curvature * outgoing.square().sum(0)
        choice = torch.stack((ordinary, constant, disconnected)).argmin(0)
        constant, disconnected = choice.eq(1), choice.eq(2)
        values[left][constant] = incoming[constant]
        values[left][constant, :fan_in] = 0.
        values[right][:, :len(incoming)][:, constant] = outgoing[:, constant]
        values[left][disconnected] = incoming[disconnected]
        values[right][:, :len(incoming)][:, disconnected] = 0.
    return values


def chain_proximal(z, penalty, curvature, penalized, pairs):
    states = [torch.zeros(len(z[left]), dtype=torch.long, device=z[left].device) for left, _, _ in pairs]
    ordinary = {name: value.masked_fill(penalized[name] & (.5 * curvature * value.square()).le(penalty), 0.)
                for name, value in z.items()}

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
            squared = .5 * curvature * value.square()
            charged = penalized[name] & ~exempt[name]
            costs[name] = torch.where(forced[name], squared,
                                     torch.where(charged, squared.clamp_max(penalty), 0.))
            values[name] = value.masked_fill(forced[name] | (charged & squared.le(penalty)), 0.)
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
        return (.5 * curvature * sum(dot(value-z[name], value-z[name]) for name, value in values.items())
                + penalty * canonical_nonzeros(values, penalized, pairs))
    return candidate if objective(candidate) <= objective(ordinary) else ordinary


@torch.no_grad()
def minimize_l0(reference, factors, penalty, penalized, options, *, pair=None, fixed=None, inverses=None):
    """Penalized proximal gradient; a foldable pair has an exact three-way group proximal map."""
    def product(values):
        return {name: factors[name][1] @ value @ factors[name][0] for name, value in values.items()}

    def damage_at(values, gradients):
        return .5 * sum(dot(values[name] - reference[name], gradient) for name, gradient in gradients.items())

    def refit(values):
        refined, steps, residuals = {}, {}, {}
        for name, weight in values.items():
            a, b = factors[name]
            refined[name], steps[name], residuals[name] = support_refit(
                weight, reference[name], a, b, penalized[name],
                options.refit_iterations, options.refit_rtol,
                fixed[name] if fixed is not None else None,
                inverses[name] if inverses is not None else None)
        return refined, steps, residuals

    if options.refit_iterations and inverses is None:
        inverses = {name: tuple(torch.cholesky_inverse(torch.linalg.cholesky(factor)) for factor in (a, b))
                    for name, (a, b) in factors.items()}
    value = {name: weight.clone() for name, weight in reference.items()}
    floor = max(initial_curvature(a, b) for a, b in factors.values())
    curvature = floor
    gradient = {name: torch.zeros_like(weight) for name, weight in reference.items()}
    damage = 0.
    objective = penalty * canonical_nonzeros(value, penalized, pair)
    backtracks, converged = 0, False
    polished, restart = False, False
    refit_calls, total_refit_steps = 0, {name: 0 for name in value}
    dtype = next(iter(value.values())).dtype
    epsilon = 16 * torch.finfo(dtype).eps
    for iteration in range(1, options.max_iterations + 1):
        curvature = max(floor, curvature / 2)
        # Before declaring a fixed point, retry the full structured prox at the initial step scale.
        if restart:
            curvature = floor
        restart = False
        started_at_floor = curvature == floor
        for _ in range(options.max_backtracks):
            candidate = proximal_step({name: weight - gradient[name] / curvature for name, weight in value.items()},
                                      penalty, curvature, penalized, pair, fixed)
            next_gradient = product({name: weight - reference[name] for name, weight in candidate.items()})
            step = {name: weight - value[name] for name, weight in candidate.items()}
            remainder = sum(dot(step[name], next_gradient[name] - gradient[name]) for name in value)
            bound = curvature * sum(dot(delta, delta) for delta in step.values())
            next_damage = damage_at(candidate, next_gradient)
            next_objective = next_damage + penalty * canonical_nonzeros(candidate, penalized, pair)
            if (math.isfinite(next_objective) and next_objective <= objective and
                    remainder <= bound + epsilon * (abs(remainder) + abs(bound))):
                break
            curvature *= 2
            backtracks += 1
        else:
            raise RuntimeError("L0 proximal backtracking could not find a descent step")
        same_support = all(torch.equal(candidate[name].eq(0), weight.eq(0)) for name, weight in value.items())
        small_step = (sum(dot(delta, delta) for delta in step.values()) <= options.rtol**2 *
                      max(sum(dot(weight, weight) for weight in value.values()), torch.finfo(dtype).tiny))
        value, gradient, damage, objective = candidate, next_gradient, next_damage, next_objective
        if not same_support:
            polished = False
        if same_support and small_step:
            if options.refit_iterations and not polished:
                refined, steps, _ = refit(value)
                refit_calls += 1
                for name, count in steps.items():
                    total_refit_steps[name] += count
                refined_gradient = product({name: weight - reference[name] for name, weight in refined.items()})
                refined_damage = damage_at(refined, refined_gradient)
                refined_objective = refined_damage + penalty * canonical_nonzeros(refined, penalized, pair)
                if math.isfinite(refined_objective) and refined_objective <= objective:
                    value, gradient, damage, objective = refined, refined_gradient, refined_damage, refined_objective
                polished, restart = True, True
                continue
            if not started_at_floor:
                restart = True
                continue
            converged = True
            break
    refit_steps, refit_residual, refit_accepted = {}, {}, False
    if options.refit_iterations:
        refined, refit_steps, refit_residual = refit(value)
        refit_calls += 1
        for name, count in refit_steps.items():
            total_refit_steps[name] += count
        refined_gradient = product({name: weight - reference[name] for name, weight in refined.items()})
        refined_damage = damage_at(refined, refined_gradient)
        refined_objective = refined_damage + penalty * canonical_nonzeros(refined, penalized, pair)
        if math.isfinite(refined_objective) and refined_objective <= objective:
            value, damage, objective = refined, refined_damage, refined_objective
            refit_accepted = True
    return value, {"iterations": iteration, "proximal_converged": converged,
                   "backtracks": backtracks, "lipschitz": curvature,
                   "predicted_fisher_damage": damage, "penalized_objective": objective,
                   "nonzeros": canonical_nonzeros(value, penalized, pair), "penalty": penalty,
                   "refit_iterations": refit_steps, "refit_relative_residual": refit_residual,
                   "refit_accepted": refit_accepted, "refit_calls": refit_calls,
                   "refit_total_iterations": total_refit_steps}
