from __future__ import annotations

import math
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from pruning.experiments.opt_cpu import linear_layers


def estimate_b_covariance(model, sequences, names, max_tokens, seed,
                          layers_count, layer_index=None):
    modules = {(index, name): linear_layers(layer)[name]
               for index, layer in enumerate(model.model.decoder.layers[:layers_count])
               for name in names if layer_index is None or index == layer_index}
    covariance = {key: torch.zeros((module.out_features, module.out_features),
                                    dtype=torch.float64)
                  for key, module in modules.items()}
    counts = {key: 0 for key in modules}
    captured, handles = {}, []
    for key, module in modules.items():
        def capture(_module, _inputs, output, name=key):
            captured[name] = output
            output.retain_grad()
        handles.append(module.register_forward_hook(capture))
    required = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embed_handle = model.model.decoder.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: output.detach().requires_grad_(True))
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches = DataLoader(TensorDataset(torch.cat(sequences)), batch_size=1)
    quota = max(1, math.ceil(max_tokens / len(batches)))
    started = time.perf_counter()
    try:
        with torch.enable_grad():
            for (ids,) in batches:
                if min(counts.values()) >= max_tokens:
                    break
                captured.clear()
                ids = ids.to(device)
                logits = model(ids, use_cache=False).logits[:, :-1].float()
                probabilities = torch.softmax(logits.detach(), -1)
                labels = torch.multinomial(
                    probabilities.reshape(-1, probabilities.shape[-1]), 1,
                    generator=generator).reshape(logits.shape[:-1])
                torch.log_softmax(logits, -1).gather(
                    -1, labels.unsqueeze(-1)).mean().backward()
                for key, output in captured.items():
                    gradient = output.grad.reshape(*ids.shape, -1)[:, :-1].reshape(
                        -1, output.shape[-1]).float()
                    gradient.mul_(labels.numel())
                    count = min(quota, max_tokens - counts[key], len(gradient))
                    positions = torch.linspace(
                        0, len(gradient) - 1, count, device=device).long()
                    selected = gradient[positions].double().cpu()
                    covariance[key].addmm_(selected.T, selected)
                    counts[key] += count
    finally:
        embed_handle.remove()
        for handle in handles:
            handle.remove()
        for parameter, value in zip(model.parameters(), required):
            parameter.requires_grad_(value)
    return covariance, counts, time.perf_counter() - started


def inverse_factors(covariance, counts, damping_ratio):
    factors = {}
    for key in tuple(covariance):
        factor = covariance.pop(key)
        factor.div_(counts[key])
        damping = damping_ratio * max(
            float(factor.diag().mean()), torch.finfo(factor.dtype).eps)
        factor.diagonal().add_(damping)
        cholesky, info = torch.linalg.cholesky_ex(factor)
        if int(info):
            raise RuntimeError(f"B factor is not positive definite: {key}")
        factors[key] = (factor, torch.cholesky_inverse(cholesky))
    return factors
