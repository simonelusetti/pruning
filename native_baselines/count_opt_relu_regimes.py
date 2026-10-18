#!/usr/bin/env python3
"""Count observed ReLU regimes for every OPT feed-forward neuron.

The default arguments reproduce the calibration inputs used by the fair
SparseGPT run: 128 C4 samples of length 2048, sampled with seed 0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_cpu_close import DATASET_CACHE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int,
                        default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--output", type=Path,
        default=Path("native_baselines/results_fair_sparsegpt/relu_regimes.json"),
    )
    return parser.parse_args()


def cached_c4_tokens(tokenizer, nsamples: int, seqlen: int,
                     seed: int) -> list[torch.Tensor]:
    """Recreate calibration_tokens from the cached C4 Arrow shards.

    Loading through ``load_dataset`` ties the cache fingerprint to the original
    absolute workspace path.  Reading the already-built Arrow shards directly
    preserves their row order and therefore the original seed-0 sampling.
    """
    files = sorted(DATASET_CACHE.glob(
        "allenai___c4/*/0.0.0/*/c4-train-*.arrow"))
    if not files:
        raise FileNotFoundError(f"no cached C4 Arrow shards below {DATASET_CACHE}")
    dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in files])
    rng = random.Random(seed)
    samples = []
    while len(samples) < nsamples:
        text = dataset[rng.randrange(len(dataset))]["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[:, start:start + seqlen])
    return samples


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).eval()
    model.config.use_cache = False
    samples = cached_c4_tokens(
        tokenizer, args.nsamples, args.seqlen, args.seed)

    layers = model.model.decoder.layers
    seen_positive = [torch.zeros(layer.fc1.out_features, dtype=torch.bool)
                     for layer in layers]
    seen_nonpositive = [torch.zeros(layer.fc1.out_features, dtype=torch.bool)
                        for layer in layers]
    minima = [torch.full((layer.fc1.out_features,), float("inf"))
              for layer in layers]
    maxima = [torch.full((layer.fc1.out_features,), float("-inf"))
              for layer in layers]

    handles = []
    for layer_index, layer in enumerate(layers):
        def record(_module, _inputs, output, index=layer_index):
            values = output.detach().reshape(-1, output.shape[-1]).float().cpu()
            seen_positive[index] |= (values > 0).any(dim=0)
            seen_nonpositive[index] |= (values <= 0).any(dim=0)
            minima[index] = torch.minimum(minima[index], values.amin(dim=0))
            maxima[index] = torch.maximum(maxima[index], values.amax(dim=0))

        handles.append(layer.fc1.register_forward_hook(record))

    try:
        for sample_index, input_ids in enumerate(samples, 1):
            model(input_ids, use_cache=False)
            if sample_index % 8 == 0 or sample_index == len(samples):
                print(f"processed {sample_index}/{len(samples)} sequences", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    layer_rows = []
    total_active = total_inactive = total_switching = total_unseen = 0
    for index, layer in enumerate(layers):
        active = seen_positive[index] & ~seen_nonpositive[index]
        inactive = seen_nonpositive[index] & ~seen_positive[index]
        switching = seen_positive[index] & seen_nonpositive[index]
        unseen = ~seen_positive[index] & ~seen_nonpositive[index]
        width = layer.fc1.out_features
        row = {
            "layer": index,
            "neurons": width,
            "always_active": int(active.sum()),
            "always_inactive": int(inactive.sum()),
            "switching": int(switching.sum()),
            "unseen": int(unseen.sum()),
            "always_active_fraction": float(active.float().mean()),
            "always_inactive_fraction": float(inactive.float().mean()),
            "switching_fraction": float(switching.float().mean()),
            "smallest_always_active_preactivation": (
                float(minima[index][active].min()) if active.any() else None),
            "largest_always_inactive_preactivation": (
                float(maxima[index][inactive].max()) if inactive.any() else None),
        }
        layer_rows.append(row)
        total_active += row["always_active"]
        total_inactive += row["always_inactive"]
        total_switching += row["switching"]
        total_unseen += row["unseen"]

    total = total_active + total_inactive + total_switching + total_unseen
    result = {
        "model": args.model,
        "activation_function": model.config.activation_function,
        "dataset": "allenai/c4 train shard 00000-of-01024",
        "seed": args.seed,
        "sequences": args.nsamples,
        "sequence_length": args.seqlen,
        "observed_tokens": args.nsamples * args.seqlen,
        "classification": {
            "always_active": "fc1 preactivation > 0 for every observed token",
            "always_inactive": "fc1 preactivation <= 0 for every observed token",
            "switching": "both > 0 and <= 0 were observed",
        },
        "total": {
            "neurons": total,
            "always_active": total_active,
            "always_inactive": total_inactive,
            "switching": total_switching,
            "unseen": total_unseen,
            "always_active_fraction": total_active / total,
            "always_inactive_fraction": total_inactive / total,
            "switching_fraction": total_switching / total,
        },
        "layers": layer_rows,
        "seconds": time.perf_counter() - started,
        "scope_note": (
            "Observed on the SparseGPT calibration sample; this does not certify "
            "the activation regime over every valid model input."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
