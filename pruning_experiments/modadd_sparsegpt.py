#!/usr/bin/env python3
"""Run SparseGPT on the maintained modular-addition checkpoint."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))

from nnprune.experiments import (checkpoint_sha256, load_benchmark,
                                 split_validation)  # noqa: E402
from native_baselines.result_artifacts import write_json  # noqa: E402
from native_baselines.sparsegpt import sparsegpt_matrix  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-spacing", type=float, default=.02)
    parser.add_argument("--fine-until", type=float, default=.2)
    parser.add_argument("--coarse-spacing", type=float, default=.05)
    parser.add_argument("--max-sparsity", type=float, default=.5)
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "native_baselines/results_modadd_grid/sparsegpt.json")
    return parser.parse_args(argv)


def schedule(fine_spacing, fine_until, coarse_spacing, maximum):
    fine = torch.arange(fine_spacing, fine_until + fine_spacing / 2,
                        fine_spacing, dtype=torch.float64).tolist()
    start = math.ceil((fine_until + 1e-9) / coarse_spacing) * coarse_spacing
    coarse = torch.arange(start, maximum + coarse_spacing / 2,
                          coarse_spacing, dtype=torch.float64).tolist()
    return ([round(value, 10) for value in fine]
            + [round(value, 10) for value in coarse if value > fine_until + 1e-9])


def calibration_inputs(adapter, state, count):
    X = adapter.data["train"][0]
    generator = torch.Generator(device=X.device).manual_seed(7)
    X = X[torch.randperm(len(X), generator=generator, device=X.device)[:count]]
    parameters = adapter.effective(state)
    _, cache = adapter.forward_full(parameters, state.struct, X)
    d, heads, head_width = adapter.d, adapter.H, adapter.dh
    return {
        "W_Q": cache["x0"].reshape(-1, d),
        "W_K": cache["x0"].reshape(-1, d),
        "W_V": cache["x0"].reshape(-1, d),
        "W_O": cache["z"].permute(0, 2, 1, 3).reshape(-1, heads * head_width),
        "W_in": cache["x1"].reshape(-1, d),
        "W_out": cache["act"].reshape(-1, adapter.dm),
    }


def matrix_view(adapter, state, name):
    weight = adapter.effective(state)[name]
    if name in ("W_Q", "W_K", "W_V"):
        return weight.permute(0, 2, 1).reshape(adapter.H * adapter.dh, adapter.d)
    if name == "W_O":
        return weight.reshape(adapter.H * adapter.dh, adapter.d).T
    return weight


def write_matrix(adapter, state, name, weight, pruned):
    live = ~pruned
    if name in ("W_Q", "W_K", "W_V"):
        shape = (adapter.H, adapter.dh, adapter.d)
        state.params[name] = weight.reshape(shape).permute(0, 2, 1).contiguous()
        state.masks[name] = live.reshape(shape).permute(0, 2, 1).contiguous()
    elif name == "W_O":
        state.params[name] = weight.T.reshape(adapter.H, adapter.dh, adapter.d).contiguous()
        state.masks[name] = live.T.reshape(adapter.H, adapter.dh, adapter.d).contiguous()
    else:
        state.params[name] = weight.contiguous()
        state.masks[name] = live.contiguous()


@torch.no_grad()
def sparsegpt_state(adapter, initial, inputs, sparsity):
    state = initial.clone()
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"):
        weight = matrix_view(adapter, state, name)
        reconstructed, pruned = sparsegpt_matrix(
            weight, inputs[name].float().T @ inputs[name].float(), sparsity,
            percdamp=.01, blocksize=128)
        write_matrix(adapter, state, name, reconstructed, pruned)
    adapter.canonicalize(state)
    return state


def sparsegpt_curve(adapter, initial, sparsities, calibration, args):
    k0 = adapter.k_eff(initial)
    reference = adapter.per_sample(initial, "valB").double()
    base_loss = float(reference.mean())
    base_test = adapter.metrics(initial, "test")
    started = time.perf_counter()
    rows = [{"step": 0, "target_sparsity": 0.0, "removed": 0,
             "frac_removed": 0.0, "val_loss": base_loss, "val_delta": 0.0,
             "val_se": 0.0, "test_loss": base_test["loss"],
             "test_acc": base_test["acc"], "seconds": 0.0}]
    stop_index = None
    common = {
        "result_schema": 3, "model": "modadd_p59_one_layer_transformer",
        "dataset": "a + b mod 59", "checkpoint": "checkpoints/modadd_seed0.pt",
        "checkpoint_sha256": checkpoint_sha256("modadd"), "method": "SparseGPT",
        "protocol": {"split_seed": 0, "val_a": 435, "val_b": 435, "test": 871,
                     "rho": .05, "n_sigma": 2.0, "calibration_split": "train",
                     "calibration_count": min(args.calibration,
                                              len(adapter.data["train"][0])),
                     "calibration_seed": 7, "percdamp": .01, "blocksize": 128,
                     "schedule": sparsities},
    }
    for step, sparsity in enumerate(sparsities, 1):
        state = sparsegpt_state(adapter, initial, calibration, sparsity)
        val = adapter.per_sample(state, "valB").double()
        difference = val - reference
        delta = float(difference.mean())
        se = float(difference.std(unbiased=True) / math.sqrt(len(difference)))
        test = adapter.metrics(state, "test")
        removed = k0 - adapter.k_eff(state)
        row = {"step": step, "target_sparsity": sparsity, "removed": removed,
               "frac_removed": removed / k0, "val_loss": float(val.mean()),
               "val_delta": delta, "val_se": se, "test_loss": test["loss"],
               "test_acc": test["acc"], "seconds": time.perf_counter() - started}
        rows.append(row)
        if stop_index is None and delta > 2 * se + .05 * base_loss:
            stop_index = len(rows) - 2
        write_json(args.output, {**common, "status": "running", "history": rows,
                                 "stop_index": stop_index})
        print(f"SparseGPT target {100*sparsity:.0f}%: "
              f"removed {100*row['frac_removed']:.2f}%, "
              f"test loss {row['test_loss']/base_test['loss']:.3f}x", flush=True)
    return {**common, "status": "complete", "history": rows,
            "stop_index": stop_index,
            "stopping_point": rows[stop_index] if stop_index is not None else rows[-1]}


def main(argv=None):
    args = parse_args(argv)
    if args.calibration < 1 or args.threads < 1:
        raise ValueError("calibration and thread counts must be positive")
    if not 0 < args.fine_spacing <= args.fine_until < args.max_sparsity < 1:
        raise ValueError("invalid fine sparsity range")
    if not 0 < args.coarse_spacing <= args.max_sparsity:
        raise ValueError("invalid coarse spacing")
    sparsities = schedule(args.fine_spacing, args.fine_until,
                          args.coarse_spacing, args.max_sparsity)
    torch.set_num_threads(args.threads)
    adapter, initial = load_benchmark("modadd", args.device)
    adapter.data = split_validation(adapter.data)
    calibration = calibration_inputs(adapter, initial, args.calibration)
    artifact = sparsegpt_curve(adapter, initial, sparsities, calibration, args)
    write_json(args.output, artifact)
    print(f"Saved {args.output}", flush=True)
