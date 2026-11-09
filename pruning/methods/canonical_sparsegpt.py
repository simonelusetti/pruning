from __future__ import annotations

import math
import time
import torch
from pruning.baselines.language import collect_stats, HessianStats
from pruning.baselines.sparsegpt import sparsegpt_matrix
from pruning.models.opt import capture_first_inputs, forward_layer, linear_layers
from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain


def completion_scales(layer, power):
    incoming = layer.fc1.weight.ne(0).float().mean(1, keepdim=True)
    outgoing = layer.fc2.weight.ne(0).float().mean(0, keepdim=True)
    return {"fc1": incoming.pow(power), "fc2": outgoing.pow(power)}


@torch.no_grad()
def cleanup(layer, masks, index):
    prefix = f"model.decoder.layers.{index}."
    first, bias, second = (masks[prefix + name] for name in ("fc1.weight", "fc1.bias", "fc2.weight"))
    before = int(first.sum() + bias.sum() + second.sum())
    previous = first.all(1) & bias & second.all(0)
    canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))
    dead = layer.fc1.weight.eq(0).all(1) & layer.fc1.bias.eq(0) & layer.fc2.weight.eq(0).all(0)
    first |= dead[:, None]
    bias |= dead
    second |= dead[None, :]
    savings = int(first.sum() + bias.sum() + second.sum()) - before
    return savings, int((dead & ~previous).sum())


class CanonicalSparseGPT:
    uses_selection = False

    def __init__(self, problem, options):
        if (not math.isfinite(options.completion_power) or options.completion_power < 0 or
                options.percdamp < 0 or options.blocksize < 1):
            raise ValueError("invalid canonical SparseGPT settings")
        if options.get("modules", "all") != "all":
            raise ValueError("canonical SparseGPT currently prunes all transformer linear modules")
        self.options = options
        problem.extra["pruned_coordinates"] = {
            f"model.decoder.layers.{index}.{name}.weight": torch.zeros_like(module.weight, dtype=torch.bool)
            for index, layer in enumerate(problem.model.model.decoder.layers[:problem.layers_count])
            for name, module in linear_layers(layer).items() if name in problem.names}
        for index, layer in enumerate(problem.model.model.decoder.layers[:problem.layers_count]):
            problem.extra["pruned_coordinates"][f"model.decoder.layers.{index}.fc1.bias"] = torch.zeros_like(
                layer.fc1.bias, dtype=torch.bool)

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        options = self.options
        layers = problem.model.model.decoder.layers[:problem.layers_count]
        scales = [completion_scales(layer, options.completion_power) for layer in layers]
        hidden, kwargs = capture_first_inputs(problem.model, problem.calibration)
        masks = problem.extra["pruned_coordinates"]
        history, direct, savings, neurons = [], 0, 0, 0
        started = time.perf_counter()
        for layer_index, layer in enumerate(layers):
            layer_started = time.perf_counter()
            modules, stats = collect_stats(layer, hidden, kwargs, "sparsegpt")
            records = {}
            for name, module in modules.items():
                if name == "fc2":
                    # fc1 pruning and exact cleanup change the features available for compensation.
                    stats[name] = HessianStats(module)
                    handle = module.register_forward_pre_hook(
                        lambda _m, args: stats["fc2"].add(args[0].detach().cpu()))
                    try:
                        forward_layer(layer, hidden, kwargs, problem.sampling.activation_batch_size)
                    finally:
                        handle.remove()
                key = f"model.decoder.layers.{layer_index}.{name}.weight"
                values, deleted = sparsegpt_matrix(
                    module.weight, stats[name].H, target, options.percdamp, options.blocksize,
                    score_scale=scales[layer_index].get(name), fixed_mask=module.weight.eq(0))
                fresh = deleted & module.weight.ne(0) & ~masks[key]
                number = int(fresh.sum())
                masks[key] |= fresh
                module.weight.copy_(values)
                direct += number
                records[name] = {"direct_deletions": number}
                multiplier = scales[layer_index].get(name)
                if multiplier is not None:
                    records[name].update(completion_scale_min=float(multiplier.min()),
                                         completion_scale_mean=float(multiplier.mean()),
                                         completion_scale_max=float(multiplier.max()))
                if name in {"fc1", "fc2"}:
                    extra, killed = cleanup(layer, masks, layer_index)
                    savings += extra
                    neurons += killed
                    records[name].update(canonical_savings=extra, removed_neurons=killed)
            hidden = forward_layer(layer, hidden, kwargs, problem.sampling.activation_batch_size)
            history.append({"layer": layer_index, "modules": records,
                            "seconds": time.perf_counter() - layer_started})
            print(f"Canonical SparseGPT block {layer_index + 1}/{len(layers)}: "
                  f"{history[-1]['seconds']:.1f}s, {neurons} neurons removed", flush=True)
            progress(layer_index + 1, history, savings, time.perf_counter() - started)
        return {"layers": history, "direct_deletions": direct,
                "canonical_savings": savings, "removed_neurons": neurons}
