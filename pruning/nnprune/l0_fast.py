from __future__ import annotations

import math
import torch
from pruning.nnprune.l0 import canonical_nonzeros, dot, proximal_step


def lipschitz_estimate(a, b, iterations=12):
    values = []
    for factor in (a, b):
        vector = torch.ones(len(factor), device=factor.device, dtype=factor.dtype)
        for _ in range(iterations):
            vector = factor @ vector
            vector /= vector.norm().clamp_min(torch.finfo(vector.dtype).tiny)
        values.append(max(float(vector @ (factor @ vector)), float(factor.diag().max())))
    return max(1.01 * math.prod(values), torch.finfo(a.dtype).tiny)


def support_refit(value, reference, product, diagonal, penalized, iterations, rtol, fixed=None):
    free = value.ne(0) | ~penalized
    if fixed is not None:
        free &= ~fixed
    residual = -product(value - reference).masked_fill(~free, 0.)
    initial = float(residual.norm())
    if initial == 0:
        return value, 0, 0.
    result = value.clone()
    z = residual / diagonal.clamp_min(torch.finfo(value.dtype).tiny)
    direction, rz = z.clone(), dot(residual, z)
    completed = 0
    for completed in range(1, iterations + 1):
        h = product(direction).masked_fill(~free, 0.)
        curvature = dot(direction, h)
        if not math.isfinite(curvature) or curvature <= 0:
            break
        alpha = rz / curvature
        result.add_(direction, alpha=alpha)
        residual.sub_(h, alpha=alpha)
        if float(residual.norm()) <= rtol * initial:
            break
        z = residual / diagonal.clamp_min(torch.finfo(value.dtype).tiny)
        next_rz = dot(residual, z)
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    true_residual = product(result - reference).masked_fill(~free, 0.)
    return result, completed, float(true_residual.norm()) / initial


@torch.no_grad()
def minimize_l0(reference, factors, penalty, penalized, options, *, lipschitz, pair=None, fixed=None):
    """Penalized proximal gradient; a foldable pair has an exact three-way group proximal map."""
    def product(values):
        return {name: factors[name][1] @ value @ factors[name][0] for name, value in values.items()}

    def damage_at(values, gradients):
        return .5 * sum(dot(values[name] - reference[name], gradient) for name, gradient in gradients.items())

    value = {name: weight.clone() for name, weight in reference.items()}
    curvature = max(lipschitz.values())
    gradient = {name: torch.zeros_like(weight) for name, weight in reference.items()}
    damage = 0.
    objective = penalty * canonical_nonzeros(value, penalized, pair)
    backtracks, converged = 0, False
    dtype = next(iter(value.values())).dtype
    epsilon = 16 * torch.finfo(dtype).eps
    for iteration in range(1, options.max_iterations + 1):
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
        if same_support and small_step:
            converged = True
            break
    refit_steps, refit_residual, refit_accepted = {}, {}, False
    if options.refit_iterations:
        refined = {}
        for name, weight in value.items():
            a, b = factors[name]
            refined[name], refit_steps[name], refit_residual[name] = support_refit(
                weight, reference[name], lambda delta: b @ delta @ a,
                b.diag()[:, None] * a.diag()[None, :], penalized[name],
                options.refit_iterations, options.refit_rtol,
                fixed[name] if fixed is not None else None)
        refined_gradient = product({name: weight - reference[name] for name, weight in refined.items()})
        refined_damage = damage_at(refined, refined_gradient)
        refined_objective = refined_damage + penalty * canonical_nonzeros(refined, penalized, pair)
        if math.isfinite(refined_objective) and refined_objective <= objective:
            value, damage, objective = refined, refined_damage, refined_objective
            refit_accepted = True
    return value, {"iterations": iteration, "iht_converged_before_refit": converged,
                   "backtracks": backtracks, "lipschitz": curvature,
                   "predicted_fisher_damage": damage, "penalized_objective": objective,
                   "nonzeros": canonical_nonzeros(value, penalized, pair), "penalty": penalty,
                   "refit_iterations": refit_steps, "refit_relative_residual": refit_residual,
                   "refit_accepted": refit_accepted}
