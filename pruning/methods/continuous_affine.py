from __future__ import annotations

import math
import time
import torch
from omegaconf import OmegaConf
from pruning.nnprune.kronecker import KroneckerFisher
from pruning.nnprune.network import linear_chains, parameter_name
from pruning.nnprune.factorized_fisher import complement_update
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors
from pruning.models.opt import augmented_linear, copy_augmented


class AffineContinuous:
    uses_selection = False

    def __init__(self, problem, options):
        if not options.get("affine") or options.get("deletion_selection") != "global_obs":
            raise ValueError("Supervised continuous requires branch=continuous_global_affine")
        if (options.mu != 0 or options.b_refresh != "step" or options.deletion_unit != "weight" or
                options.get("modules", "all") != "all" or options.get("accounting_modules", "all") != "all" or
                min(options.percdamp, options.b_damp_ratio) <= 0 or not 0 < options.cg_rtol < 1 or
                options.cg_max_iterations < 1):
            raise ValueError("Supervised global OBS requires all affine modules, positive damping, and per-step factors")
        self.options = options
        self.modules = {name: module for name, module in problem.model.named_modules()
                        if type(module) is torch.nn.Linear}
        self.chains = linear_chains(problem.model) if options.canonicalization else []
        self.protected = {name for chain in self.chains for name in chain.biases[1:] if name}
        self.initial_nonzero = {name: value.detach().ne(0) for name, value in problem.model.named_parameters()
                                if name.rpartition(".")[0] in self.modules}
        problem.extra["obs_live"] = {name: torch.ones_like(augmented_linear(module), dtype=torch.bool)
                                     for name, module in self.modules.items()}
        problem.extra["pruned_coordinates"] = {name: torch.ones_like(value) for name, value in self.initial_nonzero.items()}
        problem.extra["pruned_coordinate_convention"] = "live"

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        started = time.perf_counter()
        options = self.options
        settings = OmegaConf.create(dict(
            fisher_seed=options.get("fisher_seed", problem.cfg.sampling.seed),
            fisher_batch=options.get("fisher_batch", 1), likelihood=options.get("likelihood", "auto"),
            variance=options.get("variance", 1.), causal=False,
            a_damping=options.percdamp, b_damping=options.b_damp_ratio))
        x, _ = problem.fisher_data(options.get("fisher_n", problem.cfg.benchmark.get("fisher_n")), settings.fisher_seed)
        factors = KroneckerFisher(problem.model, x, settings, modules=self.modules).factors
        weights, scores, inverses = {}, [], {}
        live = problem.extra["obs_live"]
        for name, module in self.modules.items():
            weight = augmented_linear(module).double()
            a, b = factors[name]
            ai, bi = (torch.cholesky_inverse(torch.linalg.cholesky(factor)) for factor in (a, b))
            inverses[name] = ai, bi
            local = .5 * weight.square() / bi.diag()[:, None] / ai.diag()[None, :]
            original = self.initial_nonzero[parameter_name(name, "weight")]
            if module.bias is not None:
                bias = self.initial_nonzero[parameter_name(name, "bias")]
                if parameter_name(name, "bias") in self.protected:
                    bias = torch.zeros_like(bias)
                original = torch.cat((original, bias[:, None]), 1)
            local.masked_fill_(~(live[name] & original & weight.ne(0)), torch.inf)
            weights[name] = weight
            scores.append(local.flatten())
        scores = torch.cat(scores)
        required = max(0, math.ceil(target * problem.total) - (problem.total - problem.remaining()))
        if required > int(scores.isfinite().sum()):
            raise ValueError("target exceeds the remaining eligible affine coefficients")
        selected = scores.topk(required, largest=False, sorted=False).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[selected] = True
        history, offset = [], 0
        for name, weight in weights.items():
            deleted = mask[offset:offset + weight.numel()].view_as(weight) | ~live[name]
            a, b = factors[name]
            ai, bi = inverses[name]
            delta, record = complement_update(weight, a, b, deleted, rtol=options.cg_rtol,
                                              max_iterations=options.cg_max_iterations,
                                              a_inverse=ai, b_inverse=bi)
            copy_augmented(self.modules[name], (weight + delta).masked_fill(deleted, 0.))
            live[name] &= ~deleted
            offset += weight.numel()
            history.append({"module": name, **record})
            progress(len(history), history, 0, time.perf_counter() - started)
        self.masks(problem)
        before = problem.remaining()
        parameters = dict(problem.model.named_parameters())
        for chain in self.chains:
            canonicalize_mlp_tensors([parameters[name] for name in chain.weights],
                                    [parameters[name] if name else None for name in chain.biases], chain.activations)
            for first, bias, second in zip(chain.weights, chain.biases, chain.weights[1:]):
                incoming, outgoing = parameters[first], parameters[second]
                dead = incoming.eq(0).all(1) & outgoing.eq(0).all(0)
                if bias:
                    dead &= parameters[bias].eq(0)
                live[first.rpartition(".")[0]][dead] = False
                live[second.rpartition(".")[0]][:, :outgoing.shape[1]][:, dead] = False
        self.masks(problem)
        return {"layers": history, "requested_new_deletions": required,
                "singleton_score_sum": float(scores[selected].sum()),
                "canonical_savings": before - problem.remaining(), "deletion_selection": "global_obs"}

    def masks(self, problem):
        for name, module in self.modules.items():
            live = problem.extra["obs_live"][name]
            masks = problem.extra["pruned_coordinates"]
            masks[parameter_name(name, "weight")].copy_(live[:, :module.in_features] | ~self.initial_nonzero[parameter_name(name, "weight")])
            if module.bias is not None:
                masks[parameter_name(name, "bias")].copy_(live[:, -1] | ~self.initial_nonzero[parameter_name(name, "bias")])
