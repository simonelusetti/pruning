from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from pruning.nnprune.network import Network, linear_chains, parameter_name
from pruning.nnprune.fisher_metric import Fisher
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors


class ReferenceBaseline:
    uses_selection = False

    def __init__(self, problem, options):
        self.options, self.scores, self.order = options, None, None
        self.kind = options.score
        if options.refresh not in {"frozen", "step"} or options.batch_size < 1 or (options.fisher_n is not None and options.fisher_n < 1):
            raise ValueError("Invalid baseline refresh or sample settings")
        if self.kind not in {"magnitude", "obd", "wanda", "random", "mean_ablation"}:
            raise ValueError("Unknown reference baseline")
        if problem.cfg.benchmark.family == "opt" and self.kind == "obd":
            raise ValueError("Exact diagonal OBD is supported on supervised benchmarks; OPT uses its separate AXB methods")
        if problem.cfg.benchmark.family == "opt" and self.kind == "mean_ablation":
            raise ValueError("mean_ablation uses the local unit/head catalogue or supervised MLP chains; OPT has the separate FLAP baseline")
        self.initial_nonzero = ({name: value.detach().ne(0) for name, value in problem.model.named_parameters()}
                                if hasattr(problem, "model") else {})
        if hasattr(problem, "model"):
            modules = getattr(problem, "affine_modules", lambda: {name: module for name, module in problem.model.named_modules() if type(module) is torch.nn.Linear})()
            self.pruning_parameters = [parameter_name(name, suffix) for name, module in modules.items()
                                       for suffix, _ in module.named_parameters(recurse=False)
                                       if suffix == "weight" or options.get("include_bias", False)]

    def inputs(self, problem):
        return problem.fisher_data(self.options.fisher_n, self.options.fisher_seed)

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        if self.kind == "mean_ablation":
            return self.mean_ablation(problem)
        if hasattr(problem, "adapter"):
            from pruning.nnprune.fdist_baseline import _prune_lowest
            state, adapter = problem.state, problem.adapter
            names = list(adapter.prunable)
            if self.scores is None or self.options.refresh == "step":
                self.scores = self.local_scores(problem, names)
            count = max(0, round(target * problem.total) - sum(int((~state.masks[name]).sum()) for name in names))
            count = _prune_lowest(state, self.scores, names, count)
        else:
            parameters = dict(problem.model.named_parameters())
            if self.scores is None or self.options.refresh == "step":
                self.scores = self.model_scores(problem)
            names = list(self.scores)
            masks = problem.extra.setdefault("pruned_coordinates", {name: torch.ones_like(parameters[name], dtype=torch.bool) for name in names})
            problem.extra["pruned_coordinate_convention"] = "live"
            count = max(0, round(target * problem.total) - sum(int((~mask).sum()) for mask in masks.values()))
            scores = torch.cat([self.scores[name].flatten().masked_fill(~(masks[name] & parameters[name].ne(0)).flatten(), torch.inf) for name in names])
            chosen = scores.topk(min(count, int(scores.isfinite().sum())), largest=False).indices
            deleted = torch.zeros_like(scores, dtype=torch.bool)
            deleted[chosen] = True
            offset = 0
            for name in names:
                local = deleted[offset:offset+parameters[name].numel()].view_as(parameters[name])
                parameters[name].masked_fill_(local, 0.)
                masks[name] &= ~local
                offset += parameters[name].numel()
            count = len(chosen)
        return {"kind": self.kind, "n_deleted": count, "score_refresh": self.options.refresh}

    def local_scores(self, problem, names):
        adapter, state = problem.adapter, problem.state
        if self.kind == "obd":
            from pruning.nnprune.core import PruneConfig, diag_fisher_and_grad, obd_scores
            settings = PruneConfig(seed=self.options.fisher_seed, fisher_batch=self.options.batch_size)
            with torch.enable_grad():
                fisher, gradient = diag_fisher_and_grad(adapter, state, settings)
            return obd_scores(state, {"fisher": fisher, "grad": gradient}, names)
        if self.kind == "wanda":
            norms = adapter.input_norms(state)
            return {name: state.params[name].abs() * norms[name] for name in names}
        generator = torch.Generator(device=state.params[names[0]].device).manual_seed(problem.cfg.benchmark.seed+11)
        return {name: (state.params[name].abs() if self.kind == "magnitude" else
                       torch.rand(state.params[name].shape, device=state.params[name].device, generator=generator)) for name in names}

    def model_scores(self, problem):
        modules = getattr(problem, "affine_modules", lambda: {name: module for name, module in problem.model.named_modules() if type(module) is torch.nn.Linear})()
        parameters = {parameter_name(name, suffix): value for name, module in modules.items()
                      for suffix, value in module.named_parameters(recurse=False)
                      if suffix == "weight" or self.options.get("include_bias", False)}
        if self.kind == "obd":
            network = Network(problem, canonicalization=False)
            x, _ = self.inputs(problem)
            diagonal = network.unpack(Fisher(network, x, batch_size=self.options.batch_size).diagonal())
            gradients = training_gradient(problem, network, self.options.batch_size)
            return {name: .5 * diagonal[name] * value.square() - gradients[name] * value for name, value in parameters.items()}
        if self.kind == "wanda":
            x, _ = self.inputs(problem)
            stats = input_moments(problem.model, modules, x, self.options.batch_size)
            return {name: value.abs() * (stats[name.rpartition('.')[0]][1].sqrt()[None, :] if value.ndim == 2 else math.sqrt(len(x)))
                    for name, value in parameters.items()}
        generator = torch.Generator(device=problem.device).manual_seed(problem.cfg.benchmark.seed+11)
        return {name: (value.abs().clone() if self.kind == "magnitude" else
                       torch.rand(value.shape, device=value.device, generator=generator)) for name, value in parameters.items()}

    def mean_ablation(self, problem):
        if hasattr(problem, "adapter"):
            if self.order is None:
                self.order = problem.adapter.classical_unit_order(problem.state)
            while self.order:
                unit = self.order.pop(0)
                proposed = problem.adapter.apply(problem.state, problem.adapter.mean_move(problem.state, unit))
                if proposed is not None and problem.adapter.k_eff(proposed) < problem.adapter.k_eff(problem.state):
                    problem.state = proposed
                    return {"kind": "mean_ablation", "unit": unit}
            return None
        chains = linear_chains(problem.model)
        parameters = dict(problem.model.named_parameters())
        modules = dict(problem.model.named_modules())
        x, _ = self.inputs(problem)
        downstream = {second.rpartition('.')[0]: modules[second.rpartition('.')[0]] for chain in chains for second in chain.weights[1:]}
        statistics = input_moments(problem.model, downstream, x, self.options.batch_size)
        if self.order is None:
            choices = []
            for ci, chain in enumerate(chains):
                for layer, second in enumerate(chain.weights[1:]):
                    name = second.rpartition('.')[0]
                    mean, second_moment, count = statistics[name]
                    variance = (second_moment / count - (mean / count).square()).clamp_min(0)
                    scores = variance * parameters[second].square().sum(0)
                    choices.extend((float(score), ci, layer, unit) for unit, score in enumerate(scores))
            self.order = sorted(choices)
        while self.order:
            _, ci, layer, unit = self.order.pop(0)
            chain = chains[ci]
            first, second = chain.weights[layer:layer+2]
            if not bool(parameters[second][:, unit].ne(0).any()):
                continue
            bias = chain.biases[layer+1]
            if bias is None:
                raise ValueError("Mean replacement requires a downstream folding bias")
            name = second.rpartition('.')[0]
            mean, _, count = statistics[name]
            parameters[bias].add_(parameters[second][:, unit] * (mean[unit]/count))
            parameters[first][unit] = 0
            if chain.biases[layer]:
                parameters[chain.biases[layer]][unit] = 0
            parameters[second][:, unit] = 0
            for chain in chains:
                canonicalize_mlp_tensors([parameters[name] for name in chain.weights],
                    [parameters[name] if name else None for name in chain.biases], chain.activations)
            masks = problem.extra.setdefault("pruned_coordinates", {name: torch.ones_like(value, dtype=torch.bool) for name, value in parameters.items()})
            problem.extra["pruned_coordinate_convention"] = "live"
            for name, parameter in parameters.items():
                masks[name] &= parameter.ne(0) | ~self.initial_nonzero[name]
            return {"kind": "mean_ablation", "layer": layer, "unit": unit}
        return None


@torch.no_grad()
def input_moments(model, modules, inputs, batch_size):
    statistics = {name: [module.weight.new_zeros(module.in_features, dtype=torch.float64),
                         module.weight.new_zeros(module.in_features, dtype=torch.float64), 0] for name, module in modules.items()}
    handles = []
    for name, module in modules.items():
        def capture(_module, args, key=name):
            values = args[0].detach().reshape(-1, args[0].shape[-1]).double()
            statistics[key][0].add_(values.sum(0))
            statistics[key][1].add_(values.square().sum(0))
            statistics[key][2] += len(values)
        handles.append(module.register_forward_pre_hook(capture))
    try:
        for (x,) in DataLoader(TensorDataset(inputs), batch_size=batch_size):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return statistics


def training_gradient(problem, network, batch_size):
    data = problem.train if hasattr(problem, "train") else problem.adapter.data["train"]
    result = {name: torch.zeros_like(value) for name, value in network.parameters.items()}
    with torch.enable_grad():
        values = {name: value.detach().clone().requires_grad_(True) for name, value in network.parameters.items()}
        for x, y in DataLoader(TensorDataset(*data), batch_size=batch_size):
            logits = network.logits(values, x.to(network.theta.device))
            loss = supervised_loss(logits, y.to(logits.device)) / len(data[0])
            gradients = torch.autograd.grad(loss, tuple(values.values()), allow_unused=True)
            for (name, _), gradient in zip(values.items(), gradients):
                if gradient is not None:
                    result[name] += gradient.detach()
    return result


def supervised_loss(logits, targets):
    if logits[:1].numel() == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), targets.to(logits).reshape(-1), reduction="sum")
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.long().reshape(-1), reduction="sum")
