from __future__ import annotations

import copy
import torch


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
