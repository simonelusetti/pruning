from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from pruning.nnprune.network import parameter_name


class ReferenceBaseline:
    """Uncompensated ranking controls: magnitude, random or Wanda-style |W|·‖x‖ scores.

    scope=global ranks every pruned tensor together; scope=local removes the target fraction from each tensor."""
    uses_selection = False

    def __init__(self, problem, branch):
        options = self.options = branch.options
        self.scope, self.scores = branch.scope, None
        if options.score not in {"magnitude", "wanda", "random"} or options.refresh not in {"frozen", "step"} or branch.scope not in {"local", "global"}:
            raise ValueError("reference scores are magnitude, wanda or random; refresh frozen or step; scope local or global")

    def counts(self, sizes, removed, target):
        if self.scope == "global":
            return {None: max(0, math.ceil(target * sum(sizes.values())) - sum(removed.values()))}
        return {name: max(0, math.ceil(target * size) - removed[name]) for name, size in sizes.items()}

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        if hasattr(problem, "adapter"):
            from pruning.nnprune.fdist_baseline import _prune_lowest
            state, adapter = problem.state, problem.adapter
            names = list(adapter.prunable)
            if self.scores is None or self.options.refresh == "step":
                self.scores = self.local_scores(problem, names)
            sizes = {name: state.masks[name].numel() for name in names}
            removed = {name: int((~state.masks[name]).sum()) for name in names}
            count = sum(_prune_lowest(state, self.scores, names if name is None else [name], quota)
                        for name, quota in self.counts(sizes, removed, target).items())
            return {"kind": self.options.score, "n_deleted": count}
        parameters = dict(problem.model.named_parameters())
        if self.scores is None or self.options.refresh == "step":
            self.scores = self.model_scores(problem)
        names = list(self.scores)
        masks = problem.extra.setdefault("pruned_coordinates", {name: torch.ones_like(parameters[name], dtype=torch.bool) for name in names})
        problem.extra["pruned_coordinate_convention"] = "live"
        scores = {name: self.scores[name].masked_fill(~(masks[name] & parameters[name].ne(0)), torch.inf) for name in names}
        count = 0
        for name, quota in self.counts({name: masks[name].numel() for name in names},
                                       {name: int((~masks[name]).sum()) for name in names}, target).items():
            group = names if name is None else [name]
            flat = torch.cat([scores[key].flatten() for key in group])
            chosen = flat.topk(min(quota, int(flat.isfinite().sum())), largest=False).indices
            deleted, offset = torch.zeros_like(flat, dtype=torch.bool), 0
            deleted[chosen] = True
            for key in group:
                local = deleted[offset:offset + parameters[key].numel()].view_as(parameters[key])
                parameters[key].masked_fill_(local, 0.)
                masks[key] &= ~local
                offset += parameters[key].numel()
            count += len(chosen)
        return {"kind": self.options.score, "n_deleted": count}

    def local_scores(self, problem, names):
        adapter, state = problem.adapter, problem.state
        if self.options.score == "wanda":
            norms = adapter.input_norms(state)
            return {name: state.params[name].abs() * norms[name] for name in names}
        generator = torch.Generator(device=state.params[names[0]].device).manual_seed(problem.cfg.benchmark.seed + 11)
        return {name: (state.params[name].abs() if self.options.score == "magnitude" else
                       torch.rand(state.params[name].shape, device=state.params[name].device, generator=generator)) for name in names}

    def model_scores(self, problem):
        modules = getattr(problem, "affine_modules", lambda: {name: module for name, module in problem.model.named_modules()
                                                              if type(module) is torch.nn.Linear})()
        parameters = {parameter_name(name, suffix): value for name, module in modules.items()
                      for suffix, value in module.named_parameters(recurse=False)
                      if suffix == "weight" or self.options.include_bias}
        if self.options.score == "wanda":
            x, _ = problem.fisher_data(self.options.fisher_n, self.options.fisher_seed)
            norms = input_norms(problem.model, modules, x, self.options.batch_size)
            return {name: value.abs() * (norms[name.rpartition('.')[0]][None, :] if value.ndim == 2 else math.sqrt(len(x)))
                    for name, value in parameters.items()}
        generator = torch.Generator(device=problem.device).manual_seed(problem.cfg.benchmark.seed + 11)
        return {name: (value.abs().clone() if self.options.score == "magnitude" else
                       torch.rand(value.shape, device=value.device, generator=generator)) for name, value in parameters.items()}


@torch.no_grad()
def input_norms(model, modules, inputs, batch_size):
    squares = {name: module.weight.new_zeros(module.in_features, dtype=torch.float64) for name, module in modules.items()}

    def capture(_module, args, key):
        squares[key].add_(args[0].detach().reshape(-1, args[0].shape[-1]).double().square().sum(0))
    handles = [module.register_forward_pre_hook(lambda module, args, key=name: capture(module, args, key))
               for name, module in modules.items()]
    try:
        for (x,) in DataLoader(TensorDataset(inputs), batch_size=batch_size):
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {name: value.sqrt() for name, value in squares.items()}
