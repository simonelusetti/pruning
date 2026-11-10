from __future__ import annotations

import copy
import torch


def fisher_subset(data, count, seed, device):
    generator = torch.Generator(device=data[0].device).manual_seed(seed)
    indices = torch.randperm(len(data[0]), generator=generator, device=data[0].device)
    indices = indices[:count] if count is not None else indices
    return tuple(value[indices].to(device) for value in data)


class TorchProblem:
    def __init__(self, model):
        self.model = model
        self.extra = {}
        self._selection_reference = None

    def snapshot(self):
        return {"parameters": {name: value.detach().clone()
                                for name, value in self.model.state_dict().items()},
                "structure": copy.deepcopy(self.extra)}

    def restore(self, snapshot):
        self.model.load_state_dict(snapshot["parameters"])
        self.extra = copy.deepcopy(snapshot["structure"])

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
