from __future__ import annotations

import copy
import statistics
import time
import torch
import torch.nn as nn


def fisher_subset(data, count, seed, device):
    generator = torch.Generator(device=data[0].device).manual_seed(seed)
    indices = torch.randperm(len(data[0]), generator=generator, device=data[0].device)
    indices = indices[:count] if count is not None else indices
    return tuple(value[indices].to(device) for value in data)


def profile(problem):
    """Median time of the held-out evaluation, repeated until a second has elapsed so small models are not noise."""
    times = []
    while not times or (sum(times) < 1 and len(times) < 25):
        started = time.perf_counter()
        losses = problem.losses("reporting")
        times.append(time.perf_counter() - started)
    return losses, {"seconds": statistics.median(times), "repeats": len(times),
                    "parameter_bytes": problem.parameter_bytes()}


def silent(module):
    """Output units of a linear layer that are identically zero."""
    return module.weight.eq(0).all(1) & (module.bias.eq(0) if module.bias is not None else True)


def shrink_linear(module, keep, dim):
    weight = module.weight[keep] if dim == 0 else module.weight[:, keep]
    bias = module.bias if module.bias is None or dim == 1 else module.bias[keep]
    smaller = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                        device=weight.device, dtype=weight.dtype)
    smaller.weight.copy_(weight)
    if bias is not None:
        smaller.bias.copy_(bias)
    return smaller


@torch.no_grad()
def compact_chain(container, indices):
    """Drop hidden units of a ReLU chain of linear layers that are silent or never read downstream."""
    for first, second in zip(indices, indices[1:]):
        keep = ~(container[second].weight.eq(0).all(0) | silent(container[first]))
        if not keep.all():
            container[first] = shrink_linear(container[first], keep, 0)
            container[second] = shrink_linear(container[second], keep, 1)


class TorchProblem:
    def __init__(self, model):
        self.model = model
        self.extra = {}
        self._selection_reference = None

    def snapshot(self):
        return {"parameters": {name: value.detach().clone()
                                if getattr(self, "snapshot_device", None) is None else
                                value.detach().to(self.snapshot_device, copy=True)
                                for name, value in self.model.state_dict().items()},
                "structure": copy.deepcopy(self.extra)}

    def restore(self, snapshot):
        self.model.load_state_dict(snapshot["parameters"])
        self.extra = copy.deepcopy(snapshot["structure"])

    def model_parameters(self):
        return sum(parameter.numel() for parameter in self.model.parameters())

    def parameter_bytes(self):
        return sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())

    def outside_removed(self):
        return 0

    def compact(self):
        pass

    def profile(self):
        return profile(self)

    def selection_reference(self):
        if self._selection_reference is None:
            self._selection_reference = self.losses("selection")
        return self._selection_reference

    def prepare(self, index):
        return {}

    def fisher_data(self, count, seed):
        return fisher_subset(self.train, count, seed, self.device)

    def masked_remaining(self):
        return self.total - sum(int((~mask).sum()) for mask in self.extra["pruned_coordinates"].values())
