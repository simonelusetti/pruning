#!/usr/bin/env python3
"""Run CPU-feasible, close adaptations of SparseGPT, Wanda, and FLAP on OPT.

This is deliberately separate from ``run_native.py``.  The latter launches the
authors' pinned CUDA programs; this runner keeps their pruning rules but uses a
common Hugging Face OPT model, configurable short calibration sequences, and a
bounded WikiText-2 evaluation so that the comparison can run on a CPU host.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsegpt import sparsegpt_matrix

DATASET_CACHE = Path(__file__).resolve().parent / "cache" / "datasets"
EVALUATION_CACHE = Path(__file__).resolve().parent / "cache" / "evaluation"


def linear_layers(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: child for name, child in module.named_modules()
            if isinstance(child, nn.Linear)}


def calibration_tokens(tokenizer, dataset: str, nsamples: int, seqlen: int,
                       seed: int) -> list[torch.Tensor]:
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          cache_dir=str(DATASET_CACHE))
        encoded = tokenizer(" ".join(ds["text"]), return_tensors="pt").input_ids
        rng = random.Random(seed)
        return [encoded[:, (i := rng.randint(0, encoded.shape[1] - seqlen - 1)):i + seqlen]
                for _ in range(nsamples)]

    try:
        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            cache_dir=str(DATASET_CACHE),
        )
    except ValueError:
        shards = sorted((DATASET_CACHE / "allenai___c4").glob("**/c4-train-*.arrow"))
        if not shards:
            raise
        ds = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
    rng = random.Random(seed)
    samples = []
    while len(samples) < nsamples:
        text = ds[rng.randrange(len(ds))]["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[:, start:start + seqlen])
    return samples


def wikitext_test_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    return ids if max_tokens <= 0 else ids[:, :max_tokens]


def official_test_tokens(tokenizer, dataset: str, seqlen: int) -> torch.Tensor:
    if dataset == "wikitext2":
        return wikitext_test_tokens(tokenizer, 0)
    if dataset == "ptb":
        ds = Dataset.from_file(str(EVALUATION_CACHE / "ptb_test.arrow"))
        return tokenizer(" ".join(ds["text"]), return_tensors="pt").input_ids
    ds = Dataset.from_file(str(EVALUATION_CACHE / "c4_validation.arrow"))
    ids = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids
    return ids[:, :256 * seqlen]


def wikitext_validation_sequences(tokenizer, seqlen: int, offset: int,
                                  count: int) -> list[torch.Tensor]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    end = (offset + count) * seqlen
    if ids.numel() < end:
        raise ValueError(f"validation split has {ids.numel()} tokens, need {end}")
    return [ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0)
            for i in range(offset, offset + count)]


def selection_sequences(tokenizer, dataset: str, seqlen: int, offset: int,
                        count: int, seed: int) -> list[torch.Tensor]:
    if dataset == "wikitext2":
        return wikitext_validation_sequences(tokenizer, seqlen, offset, count)
    samples = calibration_tokens(tokenizer, "c4", offset + count, seqlen, seed)
    return samples[offset:]


@torch.inference_mode()
def per_sequence_loss(model, sequences: list[torch.Tensor]) -> torch.Tensor:
    ids = torch.cat(sequences).to(next(model.parameters()).device)
    logits = model(ids, use_cache=False).logits[:, :-1].float()
    labels = ids[:, 1:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none").reshape(len(ids), -1).mean(1).cpu()


def sequence_losses(model, sequences: list[torch.Tensor]) -> list[float]:
    return per_sequence_loss(model, sequences).tolist()


class StopCapture(RuntimeError):
    pass


@torch.inference_mode()
def capture_first_layer_inputs(model, samples: list[torch.Tensor]):
    layers = model.model.decoder.layers
    original = layers[0]
    hidden, kwargs_list = [], []

    class Catcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, inp, **kwargs):
            hidden.append(inp.detach().cpu())
            kwargs_list.append({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                for k, v in kwargs.items()})
            raise StopCapture

    layers[0] = Catcher(original)
    try:
        for sample in samples:
            try:
                model(sample, use_cache=False)
            except StopCapture:
                pass
    finally:
        layers[0] = original
    return torch.cat(hidden, 0), kwargs_list


def layer_forward(layer, inputs: torch.Tensor, kwargs_list: list[dict]) -> torch.Tensor:
    outputs = torch.empty_like(inputs)
    with torch.inference_mode():
        for j in range(inputs.shape[0]):
            outputs[j:j + 1] = layer(inputs[j:j + 1], **kwargs_list[j])[0]
    return outputs


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
    metric = W.abs().float() * stats.scale.sqrt()[None, :]
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
                      percdamp: float, blocksize: int):
    inputs, kwargs_list = capture_first_layer_inputs(model, samples)
    for index, layer in enumerate(model.model.decoder.layers):
        started = time.perf_counter()
        subset, stats = collect_stats(layer, inputs, kwargs_list, method)
        for name, module in subset.items():
            if method == "wanda":
                wanda_prune(module, stats[name], sparsity)
            else:
                sparsegpt_prune(module, stats[name], sparsity, percdamp, blocksize)
        inputs = layer_forward(layer, inputs, kwargs_list)
        print(f"layer {index + 1}/{len(model.model.decoder.layers)}: "
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
def run_flap(model, samples, sparsity: float):
    inputs, kwargs_list = capture_first_layer_inputs(model, samples)
    layers = model.model.decoder.layers
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

        attn_feature = (attn.fluctuation *
                        layer.self_attn.out_proj.weight.double().square().sum(0)).square()
        attn_feature = ((attn_feature - attn_feature.mean()) /
                        attn_feature.std(unbiased=True).clamp_min(1e-30))
        head_scores = attn_feature.reshape(-1, head_width).mean(1)
        mlp_scores = mlp.fluctuation * layer.fc2.weight.double().square().sum(0)
        mlp_scores = ((mlp_scores - mlp_scores.mean()) /
                      mlp_scores.std(unbiased=True).clamp_min(1e-30))

        d = model.config.hidden_size
        head_cost = 4 * d * head_width + 3 * head_width
        unit_cost = 2 * d + 1
        items.extend(FlapItem("head", li, h, float(score), head_cost)
                     for h, score in enumerate(head_scores))
        items.extend(FlapItem("unit", li, u, float(score), unit_cost)
                     for u, score in enumerate(mlp_scores))
        means.append((attn.mean.float(), mlp.mean.float()))
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


@torch.inference_mode()
def perplexity(model, ids: torch.Tensor, seqlen: int,
               batch_size: int = 4) -> tuple[float, int, float]:
    model.eval()
    chunks = ids.shape[1] // seqlen
    total_nll, total_labels = 0.0, 0
    started = time.perf_counter()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    sequences = ids[:, :chunks * seqlen].reshape(chunks, seqlen)
    for start in range(0, chunks, batch_size):
        end = min(start + batch_size, chunks)
        batch = sequences[start:end]
        logits = model(batch, use_cache=False).logits[:, :-1].float()
        labels = batch[:, 1:]
        total_nll += float(loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)))
        total_labels += labels.numel()
        if end % 16 == 0 or end == chunks:
            print(f"evaluation {end}/{chunks}", flush=True)
    return math.exp(total_nll / total_labels), total_labels, time.perf_counter() - started


def linear_sparsity(model) -> tuple[int, int, float]:
    weights = [m.weight for layer in model.model.decoder.layers
               for m in linear_layers(layer).values()]
    zero = sum(int((w == 0).sum()) for w in weights)
    total = sum(w.numel() for w in weights)
    return zero, total, zero / total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=("dense", "wanda", "sparsegpt", "flap"))
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--calibration-dataset", choices=("c4", "wikitext2"), default=None)
    p.add_argument("--nsamples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--eval-tokens", type=int, default=32768)
    p.add_argument("--validation-sequences", type=int, default=0,
                   help="also evaluate this many WikiText-2 validation chunks")
    p.add_argument("--validation-offset", type=int, default=2,
                   help="first validation chunk; 2 matches our val-B split")
    p.add_argument("--selection-dataset", choices=("wikitext2", "c4"),
                   default="wikitext2")
    p.add_argument("--selection-seed", type=int, default=12345)
    p.add_argument("--sparsity", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--percdamp", type=float, default=.01)
    p.add_argument("--blocksize", type=int, default=128)
    p.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--output", type=Path)
    p.add_argument("--published-sparsegpt", action="store_true")
    p.add_argument("--skip-test-evaluation", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.published_sparsegpt and not args.skip_test_evaluation:
        expected = (args.method == "sparsegpt" and args.model == "facebook/opt-125m"
                    and args.nsamples == 128 and args.seqlen == 2048
                    and (args.sparsity is None or args.sparsity == .5)
                    and args.seed == 0 and args.percdamp == .01
                    and args.blocksize == 128)
        if not expected:
            raise ValueError("published SparseGPT mode requires sparsegpt, OPT-125M, "
                             "128x2048 C4 samples, seed 0, 50%, damp .01, block 128")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    calibration_dataset = args.calibration_dataset or ("wikitext2" if args.method == "flap" else "c4")
    sparsity = args.sparsity if args.sparsity is not None else (.2 if args.method == "flap" else .5)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              use_fast=not args.published_sparsegpt)
    dtype = "auto" if args.published_sparsegpt else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).eval()
    model.config.use_cache = False
    loaded = time.perf_counter()

    prune_extra = {}
    prune_seconds = 0.0
    if args.method != "dense":
        samples = calibration_tokens(tokenizer, calibration_dataset, args.nsamples,
                                     args.seqlen, args.seed)
        prune_start = time.perf_counter()
        if args.method == "flap":
            heads, units = run_flap(model, samples, sparsity)
            prune_extra = {"removed_heads": heads, "removed_mlp_units": units}
        else:
            run_weight_method(model, samples, args.method, sparsity,
                              args.percdamp, args.blocksize)
        prune_seconds = time.perf_counter() - prune_start

    if args.published_sparsegpt:
        evaluations = {}
        eval_seconds = 0.
        for dataset in ("wikitext2", "ptb", "c4"):
            ids = official_test_tokens(tokenizer, dataset, args.seqlen)
            ppl, labels, seconds = perplexity(model, ids, args.seqlen)
            evaluations[dataset] = {"perplexity": ppl, "evaluated_labels": labels,
                                    "seconds": seconds}
            eval_seconds += seconds
        ppl = evaluations["wikitext2"]["perplexity"]
        evaluated_labels = evaluations["wikitext2"]["evaluated_labels"]
    elif not args.skip_test_evaluation:
        ids = wikitext_test_tokens(tokenizer, args.eval_tokens)
        ppl, evaluated_labels, eval_seconds = perplexity(model, ids, args.seqlen)
        evaluations = {"wikitext2": {"perplexity": ppl,
                                       "evaluated_labels": evaluated_labels,
                                       "seconds": eval_seconds}}
    else:
        ppl, evaluated_labels, eval_seconds, evaluations = None, 0, 0., {}
    validation_losses = []
    if args.validation_sequences:
        validation = selection_sequences(
            tokenizer, args.selection_dataset, args.seqlen,
            args.validation_offset, args.validation_sequences,
            args.selection_seed)
        validation_losses = sequence_losses(model, validation)
    zero, total, actual_sparsity = linear_sparsity(model)
    result = {
        "status": "complete",
        "kind": ("published SparseGPT protocol, CPU-safe execution"
                 if args.published_sparsegpt else
                 "cpu-close adaptation; not an official paper reproduction"),
        "method": args.method,
        "model": args.model,
        "calibration_dataset": calibration_dataset if args.method != "dense" else None,
        "nsamples": args.nsamples if args.method != "dense" else 0,
        "seqlen": args.seqlen,
        "target_sparsity": sparsity if args.method != "dense" else 0.0,
        "linear_zero_sparsity": actual_sparsity,
        "linear_zero_parameters": zero,
        "linear_parameters": total,
        "wikitext2_perplexity": ppl,
        "evaluations": evaluations,
        "evaluated_labels": evaluated_labels,
        "selection_loss": (sum(validation_losses) / len(validation_losses)
                           if validation_losses else None),
        "selection_losses": validation_losses,
        "selection_split": ({"dataset": args.selection_dataset,
                             "offset": args.validation_offset,
                             "count": args.validation_sequences,
                             "seed": args.selection_seed}
                            if validation_losses else None),
        "load_seconds": loaded - started,
        "prune_seconds": prune_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": time.perf_counter() - started,
        "threads": args.threads,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "source_commit": ("147d2159dc4f3e9f73e47b32c04d7b3708f44436"
                          if args.published_sparsegpt else None),
        **prune_extra,
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    output = args.output or Path("native_baselines/results_cpu_close") / f"{args.method}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text + "\n")
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
