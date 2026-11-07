from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class HiddenLayer:
    live: Callable[[], torch.Tensor]
    constant: Callable[[], torch.Tensor]
    fan_in: Callable[[], torch.Tensor]
    fan_out: Callable[[], torch.Tensor]
    remove: Callable[[torch.Tensor], None]
    fold: Callable[[torch.Tensor], None]


@torch.no_grad()
def canonicalize_mlp(layers: Sequence[HiddenLayer]) -> bool:
    any_changed = False
    while True:
        changed = False
        for layer in layers:
            dead = (layer.live() & ~layer.fan_out()).nonzero().flatten()
            if dead.numel():
                layer.remove(dead)
                changed = any_changed = True
            constant = (layer.constant() & ~layer.fan_in() &
                        layer.fan_out()).nonzero().flatten()
            if constant.numel():
                layer.fold(constant)
                changed = any_changed = True
        if not changed:
            return any_changed


@torch.no_grad()
def canonicalize_mlp_tensors(weights: Sequence[torch.Tensor],
                             biases: Sequence[torch.Tensor | None],
                             activations: Sequence[Callable]) -> bool:
    if len(weights) != len(biases) or len(weights) != len(activations) + 1:
        raise ValueError("one bias per linear and one activation per hidden layer are required")
    layers = []
    for index, activation in enumerate(activations):
        incoming, outgoing = weights[index], weights[index + 1]
        incoming_bias, outgoing_bias = biases[index], biases[index + 1]

        def live(incoming=incoming, outgoing=outgoing,
                 incoming_bias=incoming_bias):
            bias_live = (incoming_bias != 0 if incoming_bias is not None else
                         torch.zeros(incoming.shape[0], dtype=torch.bool,
                                     device=incoming.device))
            return ((incoming != 0).any(1) | bias_live |
                    (outgoing != 0).any(0))

        def remove(ids, incoming=incoming, outgoing=outgoing,
                   incoming_bias=incoming_bias):
            incoming[ids] = 0
            if incoming_bias is not None:
                incoming_bias[ids] = 0
            outgoing[:, ids] = 0

        def constant(incoming=incoming, incoming_bias=incoming_bias,
                     outgoing_bias=outgoing_bias, activation=activation):
            eligible = torch.ones(incoming.shape[0], dtype=torch.bool,
                                  device=incoming.device)
            if outgoing_bias is None:
                bias = (incoming_bias if incoming_bias is not None else
                        incoming.new_zeros(incoming.shape[0]))
                eligible &= activation(bias) == 0
            return eligible

        def fold(ids, incoming=incoming, outgoing=outgoing,
                 incoming_bias=incoming_bias, outgoing_bias=outgoing_bias,
                 activation=activation, remove=remove):
            bias = (incoming_bias[ids] if incoming_bias is not None else
                    incoming.new_zeros(len(ids)))
            values = activation(bias)
            if outgoing_bias is not None:
                outgoing_bias.add_(outgoing[:, ids] @ values)
            remove(ids)

        layers.append(HiddenLayer(
            live=live,
            constant=constant,
            fan_in=lambda incoming=incoming: (incoming != 0).any(1),
            fan_out=lambda outgoing=outgoing: (outgoing != 0).any(0),
            remove=remove,
            fold=fold,
        ))
    return canonicalize_mlp(layers)


@torch.no_grad()
def canonicalize_linear_chain(linears: Sequence[nn.Linear],
                              activations: Sequence[Callable]) -> bool:
    return canonicalize_mlp_tensors(
        [linear.weight for linear in linears],
        [linear.bias for linear in linears], activations)
