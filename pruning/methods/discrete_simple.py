from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
import math
import time
import torch
from torch import nn
from pruning.nnprune.kronecker import KroneckerFisher
from pruning.nnprune.l0 import lipschitz_estimate, minimize_l0
from pruning.nnprune.network import linear_chains
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors


def parameter_name(module, suffix):
    return f"{module}.{suffix}" if module else suffix


class DiscreteSimple:
    uses_selection = False

    def __init__(self, problem, options):
        if not hasattr(problem, "model"):
            raise ValueError("L0 AXB optimization requires a PyTorch model with linear modules")
        if (options.metric != "axb" or options.b_refresh not in {"step", "frozen"} or
                min(options.a_damping, options.b_damping) < 0 or options.variance <= 0 or
                min(options.max_iterations, options.max_backtracks, options.penalty_trials, options.fisher_batch) < 1 or
                options.refit_iterations < 0 or not 0 < options.rtol < 1 or not 0 < options.refit_rtol < 1 or
                not 0 <= options.sparsity_tolerance < 1 or options.penalty_growth <= 1 or
                (options.penalty is not None and (not math.isfinite(options.penalty) or options.penalty < 0)) or
                (options.fisher_n is not None and options.fisher_n < 1) or
                options.likelihood not in {"auto", "bernoulli", "categorical", "gaussian"} or
                options.solver_dtype not in {"float32", "float64"}):
            raise ValueError("Invalid matrix-free L0 settings")
        if options.get("penalty_mode", "retune") not in {"retune", "fixed"} or not 0 < options.get("initial_sparsity", .05) < 1:
            raise ValueError("Invalid L0 penalty mode or initial sparsity")
        if options.b_refresh == "frozen" and problem.cfg.sampling.refresh != "fixed":
            raise ValueError("Frozen factors require sampling.refresh=fixed")
        problem.model.eval()
        self.options, self.cached = options, None
        self.dtype = getattr(torch, options.solver_dtype)
        discover = getattr(problem, "affine_modules", lambda: {
            name: module for name, module in problem.model.named_modules() if type(module) is nn.Linear})
        patterns = options.parameters
        patterns = [patterns] if isinstance(patterns, str) else patterns
        owners = Counter(id(parameter) for module in problem.model.modules()
                         for parameter in module.parameters(recurse=False))
        self.modules, self.penalized, self.initial_nonzero = {}, {}, {}
        masks = problem.extra.setdefault("pruned_coordinates", {})
        problem.extra["pruned_coordinate_convention"] = "live"
        for name, module in discover().items():
            penalized = torch.zeros((module.out_features, module.in_features + int(module.bias is not None)),
                                    dtype=torch.bool, device=module.weight.device)
            for suffix, parameter in module.named_parameters(recurse=False):
                key = parameter_name(name, suffix)
                selected = patterns is None or any(fnmatchcase(key, pattern) for pattern in patterns)
                if suffix == "weight":
                    penalized[:, :module.in_features] = selected
                elif options.penalize_bias:
                    penalized[:, -1] = selected
            if not bool(penalized.any()):
                continue
            if any(owners[id(parameter)] != 1 for parameter in module.parameters(recurse=False)):
                raise ValueError(f"Shared affine parameters are not supported: {name}")
            self.modules[name], self.penalized[name] = module, penalized
            for suffix, parameter in module.named_parameters(recurse=False):
                key = parameter_name(name, suffix)
                self.initial_nonzero[key] = parameter.detach().ne(0)
                masks.setdefault(key, torch.ones_like(parameter, dtype=torch.bool))
        if not self.modules:
            raise ValueError("No affine parameters selected for the L0 penalty")
        self.chains, pairs, used = [], {}, set()
        discover = getattr(problem, "linear_chains", lambda: linear_chains(problem.model))
        for chain in discover() if options.canonicalization else ():
            names = tuple(name.rpartition(".")[0] for name in chain.weights)
            if not set(names) <= self.modules.keys():
                continue
            if len(names) != 2 or used.intersection(names):
                raise ValueError("Canonical L0 currently requires disjoint two-affine chains")
            left, right = names
            if not all(bool(self.penalized[name][:, :self.modules[name].in_features].all()) for name in names):
                raise ValueError("Canonical L0 requires both complete weight matrices in each chain")
            if self.modules[right].bias is not None:
                # Constant folding may change this bias, so it cannot be promised as a deletion.
                self.penalized[right][:, -1] = False
            self.chains.append(chain)
            pairs[left] = (left, right, self.modules[left].in_features)
            used.update(names)
        self.units = [(tuple(pair[:2]), pair) for pair in pairs.values()]
        self.units += [((name,), None) for name in self.modules if name not in used]
        if options.get("permanent_deletions", False):
            problem.extra.setdefault("l0_live", {name: torch.ones_like(mask) for name, mask in self.penalized.items()})

    def curvature(self, problem):
        if self.cached is None or self.options.b_refresh == "step":
            options = self.options
            x, _ = problem.fisher_data(options.fisher_n, options.fisher_seed)
            b_inputs = torch.cat(problem.b_sequences).to(x.device) if hasattr(problem, "b_sequences") else None
            sampling = problem.cfg.sampling
            metric = KroneckerFisher(problem.model, x, options, modules=self.modules, b_inputs=b_inputs,
                                     max_b_tokens=sampling.b_tokens or None,
                                     a_batch_size=sampling.activation_batch_size if b_inputs is not None else options.fisher_batch,
                                     b_batch_size=sampling.b_batch_size if b_inputs is not None else options.fisher_batch)
            self.cached, self.scales = {}, {}
            for name, (a, b) in metric.factors.items():
                a, b = a.to(dtype=self.dtype), b.to(dtype=self.dtype)
                scales = [torch.where(factor.diag() > 0, factor.diag().clamp_min(0).sqrt(),
                                      torch.ones_like(factor.diag())) if options.standardize
                          else torch.ones_like(factor.diag()) for factor in (a, b)]
                sa, sb = scales
                self.cached[name] = a / sa[:, None] / sa[None, :], b / sb[:, None] / sb[None, :]
                self.scales[name] = sa, sb
            self.lipschitz = {name: lipschitz_estimate(a, b) for name, (a, b) in self.cached.items()}
        return self.cached

    @torch.no_grad()
    def install(self, problem, values, *, live=None, commit=False):
        masks = problem.extra["pruned_coordinates"]
        updated = ({name: mask & ~(values[name].eq(0) & self.penalized[name]) for name, mask in live.items()}
                   if live is not None else None)
        for name, value in values.items():
            module = self.modules[name]
            sa, sb = self.scales[name]
            if live is not None:
                value = value.masked_fill(~live[name], 0.)
            value = value / sb[:, None] / sa[None, :]
            module.weight.copy_(value[:, :module.in_features])
            if module.bias is not None:
                module.bias.copy_(value[:, -1])
        parameters = dict(problem.model.named_parameters())
        before = 0
        for name, module in self.modules.items():
            for suffix, parameter in module.named_parameters(recurse=False):
                key = parameter_name(name, suffix)
                removed = (parameter.eq(0) if updated is None else
                           ~(updated[name][:, :module.in_features] if suffix == "weight" else updated[name][:, -1]))
                before += int((self.initial_nonzero[key] & removed).sum())
        for chain in self.chains:
            canonicalize_mlp_tensors([parameters[name] for name in chain.weights],
                                    [parameters[name] if name else None for name in chain.biases], chain.activations)
        if updated is not None:
            for chain in self.chains:
                left, right = (name.rpartition(".")[0] for name in chain.weights)
                incoming, outgoing = self.modules[left], self.modules[right]
                dead = incoming.weight.eq(0).all(1) & outgoing.weight.eq(0).all(0)
                if incoming.bias is not None:
                    dead &= incoming.bias.eq(0)
                updated[left][dead] = False
                updated[right][:, :incoming.out_features][:, dead] = False
            if commit:
                problem.extra["l0_live"] = updated
        for name, module in self.modules.items():
            for suffix, parameter in module.named_parameters(recurse=False):
                key = parameter_name(name, suffix)
                retained = (parameter.ne(0) if updated is None else
                            updated[name][:, :module.in_features] if suffix == "weight" else updated[name][:, -1])
                masks[key].copy_(retained | ~self.initial_nonzero[key])
        after = sum(int((~mask).sum()) for name, mask in masks.items() if name in self.initial_nonzero)
        return {"direct_removed": before, "canonical_savings": after - before}

    @torch.no_grad()
    def propose(self, problem, target, index, progress, *, capture=None):
        options, started = self.options, time.perf_counter()
        remaining_before = problem.remaining()
        fixed_penalty = options.get("penalty_mode", "retune") == "fixed"
        live = problem.extra.get("l0_live") if options.get("permanent_deletions", False) else None
        fixed = {name: ~mask for name, mask in live.items()} if live is not None else None
        stored_penalty = problem.extra.get("l0_penalty") if fixed_penalty else None
        calibrate = options.penalty is None and stored_penalty is None
        if calibrate and fixed_penalty:
            target = options.get("initial_sparsity", .05)
        factors = self.curvature(problem)
        factor_seconds = time.perf_counter() - started
        reference = {name: (torch.cat((module.weight, module.bias[:, None]), 1)
                           if module.bias is not None else module.weight).to(dtype=self.dtype).clone()
                     for name, module in self.modules.items()}
        reference = {name: value * self.scales[name][1][:, None] * self.scales[name][0][None, :]
                     for name, value in reference.items()}
        if calibrate:
            if target is None:
                raise ValueError("Specify a sparsity hint or an explicit L0 penalty")
            scores = torch.cat([.5 * self.lipschitz[name] * value[self.penalized[name] & value.ne(0)].square()
                                for name, value in reference.items()])
            needed = math.ceil(target * problem.total) - (problem.total - problem.remaining())
            rank = max(1, min(len(scores), needed))
            penalty = (max(float(scores.kthvalue(rank).values), torch.finfo(self.dtype).tiny)
                       if scores.numel() else 0.)
            del scores
        else:
            penalty = options.penalty if options.penalty is not None else stored_penalty
        lower, upper, best, trials = 0., None, None, []
        for trial in range(options.penalty_trials if calibrate else 1):
            values, history, completed = {}, [], 0
            for names, pair in self.units:
                updated, record = minimize_l0(
                    {name: reference[name] for name in names}, {name: factors[name] for name in names},
                    penalty, {name: self.penalized[name] for name in names}, options,
                    lipschitz={name: self.lipschitz[name] for name in names}, pair=pair,
                    fixed={name: fixed[name] for name in names} if fixed is not None else None)
                values.update(updated)
                for name in names:
                    self.lipschitz[name] = record["lipschitz"]
                history.append({"modules": list(names), **record})
                completed += len(names)
                progress(completed, history, 0, time.perf_counter() - started)
            cleanup = self.install(problem, values, live=live)
            sparsity = (problem.total - problem.remaining()) / problem.total
            damage = sum(record["predicted_fisher_damage"] for record in history)
            summary = {"penalty": penalty, "removed_fraction": sparsity,
                       "predicted_fisher_damage": damage,
                       "penalized_objective": sum(record["penalized_objective"] for record in history), **cleanup}
            trials.append(summary)
            distance = abs(sparsity - target) if target is not None else 0.
            if best is None or (distance, damage) < best[0]:
                best = ((distance, damage), {name: value.clone() for name, value in values.items()}, history, summary)
            print(f"L0 penalty trial {trial + 1}: lambda={penalty:.6g}, removal={100*sparsity:.3f}%", flush=True)
            if not calibrate or distance <= options.sparsity_tolerance:
                break
            if sparsity < target:
                lower = penalty
                penalty = math.sqrt(lower * upper) if upper is not None else penalty * options.penalty_growth
            else:
                upper = penalty
                penalty = math.sqrt(lower * upper) if lower else penalty / options.penalty_growth
        _, values, history, summary = best
        if capture is not None:
            capture(reference, values, factors)
        cleanup = self.install(problem, values, live=live, commit=True)
        if fixed_penalty:
            problem.extra["l0_penalty"] = summary["penalty"]
            if problem.remaining() >= remaining_before:
                return None
        return {"layers": history, **summary, "penalty_trials": trials,
                "target_within_tolerance": abs(summary["removed_fraction"] - target) <= options.sparsity_tolerance if target is not None else None,
                "new_removed_percent": 100 * (remaining_before - problem.remaining()) / problem.total,
                "penalty_fixed": fixed_penalty,
                "factor_seconds": factor_seconds,
                "iht_converged_before_refit_units": sum(row["iht_converged_before_refit"] for row in history),
                "optimized_modules": len(self.modules), "canonical_chains": len(self.chains),
                "metric": "axb", **cleanup}
