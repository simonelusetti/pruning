from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
import warnings
import torch
from torch import nn
from torch.func import functional_call
from torch.fx import symbolic_trace
from .canonicalize.mlp import canonicalize_mlp_tensors


def parameter_name(module, suffix):
    return f"{module}.{suffix}" if module else suffix


@dataclass
class LinearChain:
    weights: tuple[str, ...]
    biases: tuple[str | None, ...]
    activations: tuple


def linear_chains(model):
    try:
        graph = symbolic_trace(model).graph
    except (RuntimeError, TypeError, ValueError, torch.fx.proxy.TraceError) as error:
        warnings.warn(f"Automatic canonicalization unavailable: {error}")
        return []
    modules = dict(model.named_modules())
    calls = Counter(node.target for node in graph.nodes if node.op == "call_module")
    owners = Counter(id(parameter) for module in modules.values()
                     for parameter in module.parameters(recurse=False))
    activations = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.Identity)
    functions = {torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.relu,
                 torch.nn.functional.gelu, torch.nn.functional.silu,
                 torch.nn.functional.leaky_relu}
    pairs = {}
    for first in graph.nodes:
        if first.op != "call_module" or type(modules[first.target]) is not nn.Linear:
            continue
        if len(first.users) != 1:
            continue
        activation = next(iter(first.users))
        if activation.op == "call_module" and isinstance(modules[activation.target], activations):
            function = modules[activation.target]
        elif activation.op == "call_function" and activation.target in functions:
            function = partial(activation.target, **activation.kwargs)
        elif activation.op == "call_method" and activation.target in {"relu", "sigmoid", "tanh"}:
            function = getattr(torch, activation.target)
        else:
            continue
        if len(activation.users) != 1:
            continue
        second = next(iter(activation.users))
        if second.op != "call_module" or type(modules[second.target]) is not nn.Linear:
            continue
        left, right = modules[first.target], modules[second.target]
        foldable = right.bias is not None or (left.bias is None and
                    bool(function(left.weight.new_zeros(1)).eq(0).all()))
        if (not foldable or left.out_features != right.in_features or
                any(calls[node.target] != 1 or
                    any(owners[id(parameter)] != 1 for parameter in modules[node.target].parameters())
                    for node in (first, second))):
            continue
        pairs[first.target] = second.target, function
    chains = []
    successors = {name for name, _ in pairs.values()}
    for name in pairs:
        if name in successors:
            continue
        weights, biases, functions = [], [], []
        while True:
            weights.append(f"{name}.weight")
            biases.append(f"{name}.bias" if modules[name].bias is not None else None)
            if name not in pairs:
                break
            name, function = pairs[name]
            functions.append(function)
        chains.append(LinearChain(tuple(weights), tuple(biases), tuple(functions)))
    return chains


class Network:
    """One parameter/mask interface for nn.Module models and legacy functional models."""
    def __init__(self, problem, canonicalization=True):
        self.problem = problem
        if hasattr(problem, "model"):
            problem.model.eval()
            self.parameters = dict(problem.model.named_parameters())
            self.masks = problem.extra.get("pruned_coordinates")
            if self.masks is None:
                self.masks = {name: torch.ones_like(value, dtype=torch.bool)
                              for name, value in self.parameters.items()}
            buffers = dict(problem.model.named_buffers())
            self.forward = lambda values, x: functional_call(problem.model, (values, buffers), (x,))
            discover = getattr(problem, "linear_chains", lambda: linear_chains(problem.model))
            self.chains = discover() if canonicalization else []
        else:
            state, adapter = problem.state, problem.adapter
            self.parameters, self.masks = state.params, state.masks
            self.forward = lambda values, x: adapter.logits(values, state.struct, x)
            self.chains = adapter.linear_chains(state) if canonicalization else []
        self.offsets, start = {}, 0
        for name, value in self.parameters.items():
            self.offsets[name] = start, start + value.numel()
            start += value.numel()
        self.theta = torch.cat([value.detach().flatten() for value in self.parameters.values()]).double()
        self.live = torch.cat([self.masks[name].flatten() for name in self.parameters])

    def logits(self, values, x):
        values = {name: value * self.masks[name] for name, value in values.items()}
        result = self.forward(values, x)
        return result.logits if hasattr(result, "logits") else result

    def indices(self, name):
        start, stop = self.offsets[name]
        return torch.arange(start, stop, device=self.theta.device).view_as(self.parameters[name])

    def unpack(self, values):
        return {name: values[start:stop].view_as(self.parameters[name]).to(self.parameters[name])
                for name, (start, stop) in self.offsets.items()}

    @torch.no_grad()
    def canonicalize(self, parameters, masks):
        for name in parameters:
            parameters[name].masked_fill_(~masks[name], 0.)
        for chain in self.chains:
            canonicalize_mlp_tensors([parameters[name] for name in chain.weights],
                [parameters[name] if name else None for name in chain.biases], chain.activations)
            for first, bias, second in zip(chain.weights, chain.biases, chain.weights[1:]):
                dead = parameters[first].eq(0).all(1) & parameters[second].eq(0).all(0)
                if bias:
                    dead &= parameters[bias].eq(0)
                    masks[bias] &= ~dead
                masks[first] &= ~dead[:, None]
                masks[second] &= ~dead[None, :]
        for name in parameters:
            parameters[name].masked_fill_(~masks[name], 0.)

    def closure(self, deleted):
        parameters = {name: value.detach().clone() for name, value in self.parameters.items()}
        masks = {name: value.clone() for name, value in self.masks.items()}
        for name, (start, stop) in self.offsets.items():
            local = deleted[(deleted >= start) & (deleted < stop)] - start
            parameters[name].view(-1)[local] = 0
            masks[name].view(-1)[local] = False
        self.canonicalize(parameters, masks)
        return int(self.live.sum()) - sum(int(mask.sum()) for mask in masks.values())

    @torch.no_grad()
    def apply(self, deleted, updated, compensation):
        values, live = self.theta.clone(), self.live.clone()
        values[updated] += compensation
        values[deleted], live[deleted] = 0., False
        parameters, masks = self.unpack(values), {
            name: live[start:stop].view_as(self.parameters[name])
            for name, (start, stop) in self.offsets.items()}
        self.canonicalize(parameters, masks)
        for name, parameter in self.parameters.items():
            parameter.copy_(parameters[name])
            self.masks[name].copy_(masks[name])
