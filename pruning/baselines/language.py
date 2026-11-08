from __future__ import annotations
from dataclasses import dataclass
import math
import time
import torch
import torch.nn as nn
from pruning.models.opt import linear_layers, capture_first_inputs as capture_first_layer_inputs, forward_layer as layer_forward
from pruning.baselines.sparsegpt import sparsegpt_matrix


class WandaStats:
    def __init__(self, layer: nn.Linear):
        self.scale = torch.zeros(layer.in_features, dtype=torch.float32)
        self.nsamples = 0

    def add(self, inp: torch.Tensor):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        x = inp.reshape(-1, inp.shape[-1]).float().T
        self.scale *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        self.scale += x.norm(p=2, dim=1).square() / self.nsamples


class HessianStats:
    def __init__(self, layer: nn.Linear):
        self.H = torch.zeros((layer.in_features, layer.in_features), dtype=torch.float32)
        self.nsamples = 0

    def add(self, inp: torch.Tensor):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        x = inp.reshape(-1, inp.shape[-1]).float().T
        self.H *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        x *= math.sqrt(2 / self.nsamples)
        self.H.addmm_(x, x.T)


def collect_stats(layer, inputs, kwargs_list, kind: str):
    subset = linear_layers(layer)
    stats = {name: (HessianStats(mod) if kind == "sparsegpt" else WandaStats(mod))
             for name, mod in subset.items()}
    handles = []
    for name, mod in subset.items():
        handles.append(mod.register_forward_pre_hook(
            lambda _mod, args, key=name: stats[key].add(args[0].detach().cpu())))
    try:
        layer_forward(layer, inputs, kwargs_list)
    finally:
        for handle in handles:
            handle.remove()
    return subset, stats


@torch.inference_mode()
def wanda_prune(module: nn.Linear, stats: WandaStats, sparsity: float):
    W = module.weight.data
    count = int(W.shape[1] * sparsity)
    if count == 0:
        return
    metric = W.abs().float() * stats.scale.to(W.device).sqrt()[None, :]
    indices = torch.argsort(metric, dim=1, stable=True)[:, :count]
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    W[mask] = 0


@torch.inference_mode()
def sparsegpt_prune(module: nn.Linear, stats: HessianStats, sparsity: float,
                    percdamp: float, blocksize: int):
    values, _ = sparsegpt_matrix(
        module.weight, stats.H, sparsity, percdamp, blocksize)
    module.weight.copy_(values)


@torch.inference_mode()
def run_weight_method(model, samples, method: str, sparsity: float,
                      percdamp: float, blocksize: int, layers_count=None):
    inputs, kwargs_list = capture_first_layer_inputs(model, samples)
    layers = model.model.decoder.layers[:layers_count]
    for index, layer in enumerate(layers):
        started = time.perf_counter()
        subset, stats = collect_stats(layer, inputs, kwargs_list, method)
        for name, module in subset.items():
            if method == "wanda":
                wanda_prune(module, stats[name], sparsity)
            else:
                sparsegpt_prune(module, stats[name], sparsity, percdamp, blocksize)
        inputs = layer_forward(layer, inputs, kwargs_list)
        print(f"layer {index + 1}/{len(layers)}: "
              f"{time.perf_counter() - started:.1f}s", flush=True)


class FlapStats:
    def __init__(self, width: int):
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.fluctuation = torch.zeros(width, dtype=torch.float64)
        self.nsamples = 0

    def add(self, inp: torch.Tensor):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        for sample in inp.double().cpu():
            new_n = self.nsamples + 1
            self.mean *= self.nsamples / new_n
            self.mean += sample.mean(0) / new_n
            if self.nsamples == 0:
                self.fluctuation.zero_()
            else:
                self.fluctuation *= (self.nsamples - 1) / self.nsamples
                # This intentionally follows released BiasGPT's in-place alias.
                self.fluctuation += (sample - self.mean).square().sum(0) / new_n
            self.nsamples = new_n

@dataclass
class FlapItem:
    kind: str
    layer: int
    index: int
    score: float
    cost: int


@torch.inference_mode()
def run_flap(model, samples, sparsity: float, layers_count=None):
    inputs, kwargs_list = capture_first_layer_inputs(model, samples)
    layers = model.model.decoder.layers[:layers_count]
    head_width = model.config.hidden_size // model.config.num_attention_heads
    items: list[FlapItem] = []
    means: list[tuple[torch.Tensor, torch.Tensor]] = []

    for li, layer in enumerate(layers):
        attn = FlapStats(layer.self_attn.out_proj.in_features)
        mlp = FlapStats(layer.fc2.in_features)
        handles = [
            layer.self_attn.out_proj.register_forward_pre_hook(
                lambda _m, args, s=attn: s.add(args[0])),
            layer.fc2.register_forward_pre_hook(
                lambda _m, args, s=mlp: s.add(args[0])),
        ]
        outputs = layer_forward(layer, inputs, kwargs_list)
        for handle in handles:
            handle.remove()

        attn_feature = (attn.fluctuation.to(layer.self_attn.out_proj.weight.device) *
                        layer.self_attn.out_proj.weight.double().square().sum(0)).square()
        attn_feature = ((attn_feature - attn_feature.mean()) /
                        attn_feature.std(unbiased=True).clamp_min(1e-30))
        head_scores = attn_feature.reshape(-1, head_width).mean(1)
        mlp_scores = mlp.fluctuation.to(layer.fc2.weight.device) * layer.fc2.weight.double().square().sum(0)
        mlp_scores = ((mlp_scores - mlp_scores.mean()) /
                      mlp_scores.std(unbiased=True).clamp_min(1e-30))

        d = model.config.hidden_size
        head_cost = 4 * d * head_width + 3 * head_width
        unit_cost = 2 * d + 1
        items.extend(FlapItem("head", li, h, float(score), head_cost)
                     for h, score in enumerate(head_scores))
        items.extend(FlapItem("unit", li, u, float(score), unit_cost)
                     for u, score in enumerate(mlp_scores))
        means.append((attn.mean.to(layer.self_attn.out_proj.weight), mlp.mean.to(layer.fc2.weight)))
        inputs = outputs
        print(f"FLAP statistics layer {li + 1}/{len(layers)}", flush=True)

    ordered = sorted(items, key=lambda x: x.score, reverse=True)
    costs = torch.tensor([item.cost for item in ordered], dtype=torch.float64)
    keep_target = (1 - sparsity) * costs.sum()
    boundary = int((costs.cumsum(0) - keep_target).abs().argmin())
    threshold = ordered[boundary].score
    remove = [item for item in items if item.score <= threshold]

    for item in remove:
        layer = layers[item.layer]
        if item.kind == "head":
            start, end = item.index * head_width, (item.index + 1) * head_width
            mean = means[item.layer][0][start:end]
            contribution = mean @ layer.self_attn.out_proj.weight[:, start:end].float().T
            if layer.self_attn.out_proj.bias is None:
                layer.self_attn.out_proj.bias = nn.Parameter(contribution.to(layer.self_attn.out_proj.weight.dtype))
            else:
                layer.self_attn.out_proj.bias.add_(contribution.to(layer.self_attn.out_proj.bias.dtype))
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                               layer.self_attn.v_proj):
                projection.weight[start:end].zero_()
                if projection.bias is not None:
                    projection.bias[start:end].zero_()
            layer.self_attn.out_proj.weight[:, start:end].zero_()
        else:
            unit = item.index
            contribution = (means[item.layer][1][unit] *
                            layer.fc2.weight[:, unit].float())
            layer.fc2.bias.add_(contribution.to(layer.fc2.bias.dtype))
            layer.fc1.weight[unit].zero_()
            layer.fc1.bias[unit].zero_()
            layer.fc2.weight[:, unit].zero_()
    return len([x for x in remove if x.kind == "head"]), len([x for x in remove if x.kind == "unit"])


class LanguageBaseline:
    uses_selection = False

    def __init__(self, method, options):
        self.method, self.cfg = method, options

    def propose(self, problem, target, index, progress):
        if self.method == "flap":
            heads, units = run_flap(problem.model, problem.calibration, target, problem.layers_count)
            return {"kind": self.method, "removed_heads": heads, "removed_mlp_units": units}
        else:
            run_weight_method(problem.model, problem.calibration, self.method, target,
                              self.cfg.percdamp, self.cfg.blocksize, problem.layers_count)
        return {"kind": self.method}
