from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class GatedGroup:
    gate: str
    up: str
    down: str

    @property
    def names(self):
        return self.gate, self.up, self.down


def gated_dead(gate, up, down):
    return gate.eq(0).all(1) | up.eq(0).all(1) | down.eq(0).all(0)


@torch.no_grad()
def canonicalize_gated(gate, up, down):
    dead = gated_dead(gate, up, down)
    gate[dead] = 0
    up[dead] = 0
    down[:, dead] = 0
    return dead
