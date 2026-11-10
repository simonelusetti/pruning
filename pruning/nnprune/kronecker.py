from __future__ import annotations

import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class KroneckerFisher:
    """Independent affine-layer Fisher blocks, estimated on one unchanged model."""
    def __init__(self, model, inputs, options, *, modules=None, b_inputs=None, max_b_tokens=None,
                 a_batch_size=None, b_batch_size=None):
        modules = ({name: module for name, module in model.named_modules() if type(module) is nn.Linear}
                   if modules is None else modules)
        self.factors = {}
        statistics = {}
        handles = []
        collecting_b = False
        b_inputs = inputs if b_inputs is None else b_inputs
        collecting_a = max_b_tokens is None and inputs.shape == b_inputs.shape and torch.equal(inputs, b_inputs)
        batch_shape = None
        for name, module in modules.items():
            width = module.in_features + int(module.bias is not None)
            a = module.weight.new_zeros((width, width), dtype=torch.float64)
            b = module.weight.new_zeros((module.out_features, module.out_features), dtype=torch.float64)
            statistics[name] = [a, b, 0, 0]
            def accumulate(gradient, key=name):
                if options.causal:
                    gradient = gradient.reshape(*batch_shape, -1)[:, :-1]
                    if max_b_tokens is not None:
                        quota = max(1, math.ceil(max_b_tokens / len(b_inputs)))
                        positions = torch.linspace(0, gradient.shape[1] - 1,
                                                   min(quota, gradient.shape[1]), device=gradient.device).long()
                        gradient = gradient.index_select(1, positions)
                gradient = gradient.detach().reshape(-1, gradient.shape[-1]).double()
                if max_b_tokens is not None:
                    gradient = gradient[:max(0, max_b_tokens - statistics[key][3])]
                statistics[key][1].addmm_(gradient.T, gradient)
                statistics[key][3] += len(gradient)
            def capture(_module, args, output, key=name, include_bias=module.bias is not None):
                if collecting_a:
                    values = args[0].detach().reshape(-1, args[0].shape[-1]).double()
                    if include_bias:
                        values = torch.cat((values, values.new_ones((len(values), 1))), 1)
                    statistics[key][0].addmm_(values.T, values)
                    statistics[key][2] += len(values)
                if collecting_b:
                    output.requires_grad_(True)
                    output.register_hook(lambda gradient: accumulate(gradient, key))
            handles.append(module.register_forward_hook(capture))
        parameters = list(model.parameters())
        required = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        generator = torch.Generator(device=inputs.device).manual_seed(options.fisher_seed)
        try:
            if not collecting_a:
                collecting_a = True
                with torch.no_grad():
                    for (x,) in DataLoader(TensorDataset(inputs), batch_size=a_batch_size or options.fisher_batch):
                        model(x)
                collecting_a = False
            collecting_b = True
            with torch.enable_grad():
                for (x,) in DataLoader(TensorDataset(b_inputs), batch_size=b_batch_size or options.fisher_batch):
                    if max_b_tokens is not None and min(values[3] for values in statistics.values()) >= max_b_tokens:
                        break
                    x = x.detach().requires_grad_(x.is_floating_point())
                    batch_shape = x.shape[:2]
                    output = model(x)
                    logits = output.logits if hasattr(output, "logits") else output
                    logits = logits[:, :-1] if options.causal else logits
                    likelihood = options.likelihood
                    if likelihood == "auto":
                        likelihood = "bernoulli" if logits[:1].numel() == 1 else "categorical"
                    if likelihood == "categorical":
                        probability = logits.detach().float().softmax(-1)
                        labels = torch.multinomial(probability.reshape(-1, probability.shape[-1]),
                                                   1, generator=generator).view(*probability.shape[:-1], 1)
                        gradient = -probability
                        gradient.scatter_add_(-1, labels, torch.ones_like(labels, dtype=gradient.dtype))
                    elif likelihood == "bernoulli":
                        probability = logits.detach().sigmoid()
                        gradient = torch.bernoulli(probability, generator=generator) - probability
                    else:
                        gradient = torch.randn(logits.shape, device=logits.device, generator=generator) / math.sqrt(options.variance)
                    if logits.requires_grad:
                        logits.backward(gradient)
        finally:
            for handle in handles:
                handle.remove()
            for parameter, flag in zip(parameters, required):
                parameter.requires_grad_(flag)
        for name, (a, b, count_a, count_b) in statistics.items():
            if not count_a or not count_b:
                raise ValueError(f"No curvature observations for {name}")
            a.div_(count_a)
            b.div_(count_b)
            for factor, damping in ((a, options.a_damping), (b, options.b_damping)):
                factor.diagonal().add_(damping * factor.diag().mean().clamp_min(torch.finfo(factor.dtype).eps))
            self.factors[name] = a, b

