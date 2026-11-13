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


def gated_proximal(z, penalty, curvature, penalized, group):
    values = {name: value.masked_fill(penalized[name] & value.square().le(2 * penalty / curvature), 0.)
              for name, value in z.items()}
    gate, up, down = (z[name] for name in group.names)
    costs = {name: torch.where(penalized[name], (.5 * curvature * value.square()).clamp_max(penalty), 0.)
             for name, value in z.items()}
    ordinary = costs[group.gate].sum(1) + costs[group.up].sum(1) + costs[group.down].sum(0)
    choice = torch.stack((ordinary, .5 * curvature * gate.square().sum(1),
                          .5 * curvature * up.square().sum(1), .5 * curvature * down.square().sum(0))).argmin(0)
    dead = choice.ne(0)
    values[group.gate][dead] = gate[dead]
    values[group.up][dead] = up[dead]
    values[group.down][:, dead] = down[:, dead]
    values[group.gate][choice.eq(1)] = 0
    values[group.up][choice.eq(2)] = 0
    values[group.down][:, choice.eq(3)] = 0
    return values
