#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning.experiments.opt_cpu import (calibration_tokens, linear_layers, run_weight_method,
                           selection_sequences, sequence_losses)
from pruning.baselines.result_artifacts import write_json


from pruning.paths import ROOT


def save(path, content):
    write_json(path, content)


def decoder_parameters(model):
    for layer in model.model.decoder.layers:
        for module in linear_layers(layer).values():
            yield module.weight
            if module.bias is not None:
                yield module.bias


def paired(values, reference):
    difference = torch.tensor(values, dtype=torch.float64) - reference
    return (float(difference.mean()),
            float(difference.std(unbiased=True)) / math.sqrt(len(difference)))


def schedule(fine_spacing, fine_until, coarse_spacing, maximum):
    values = []
    value = fine_spacing
    while value <= fine_until + 1e-9:
        values.append(round(value, 10))
        value += fine_spacing
    value = math.ceil((fine_until + 1e-9) / coarse_spacing) * coarse_spacing
    while value <= maximum + 1e-9:
        if value > fine_until + 1e-9:
            values.append(round(value, 10))
        value += coarse_spacing
    return values


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-spacing", type=float, default=.02)
    parser.add_argument("--fine-until", type=float, default=.2)
    parser.add_argument("--coarse-spacing", type=float, default=.05)
    parser.add_argument("--max-sparsity", type=float, default=.5)
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=ROOT /
                        "outputs/manual/sparsegpt_curve.json")
    args = parser.parse_args(argv)
    if not 0 < args.fine_spacing <= args.fine_until < args.max_sparsity < 1:
        parser.error("invalid fine sparsity range")
    if not 0 < args.coarse_spacing <= args.max_sparsity:
        parser.error("invalid coarse spacing")

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    calibration = calibration_tokens(tokenizer, "c4", 128, 2048, 0)
    validation = selection_sequences(tokenizer, "c4", 2048, 2, 4, 12345)

    dense = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", dtype="auto").eval()
    dense.config.use_cache = False
    reference = torch.tensor(sequence_losses(dense, validation), dtype=torch.float64)
    base_loss = float(reference.mean())
    denominator = sum(parameter.numel() for parameter in decoder_parameters(dense))
    del dense

    rows = [{"target_sparsity": 0., "removed_percent": 0.,
             "loss_damage_percent": 0., "selection_loss": base_loss,
             "selection_delta": 0., "selection_se": 0., "admissible": True}]
    stop_index = None
    common = {
        "result_schema": 3, "benchmark": "opt_published",
        "method": "SparseGPT", "model": "facebook/opt-125m",
        "calibration": {"dataset": "c4", "nsamples": 128, "seqlen": 2048,
                        "seed": 0},
        "selection": {"dataset": "c4", "seed": 12345, "offset": 2,
                      "count": 4, "rho": args.rho, "n_sigma": args.n_sigma},
        "protocol": {"percdamp": .01, "blocksize": 128,
                     "source_commit": "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
                     "dtype": "float16"},
        "parameter_denominator": denominator,
    }
    for sparsity in schedule(args.fine_spacing, args.fine_until,
                             args.coarse_spacing, args.max_sparsity):
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m", dtype="auto").eval()
        model.config.use_cache = False
        run_weight_method(model, calibration, "sparsegpt", sparsity, .01, 128)
        values = sequence_losses(model, validation)
        delta, se = paired(values, reference)
        removed = sum(int((parameter == 0).sum())
                      for parameter in decoder_parameters(model))
        admissible = delta <= args.n_sigma * se + args.rho * base_loss
        rows.append({"target_sparsity": sparsity,
                     "removed_percent": 100 * removed / denominator,
                     "loss_damage_percent": 100 * (sum(values) / len(values) / base_loss - 1),
                     "selection_loss": sum(values) / len(values),
                     "selection_delta": delta, "selection_se": se,
                     "admissible": admissible})
        if stop_index is None and not admissible:
            stop_index = len(rows) - 2
        save(args.output, {**common, "status": "running",
                           "stop_index": stop_index, "curve": rows})
        print(f"SparseGPT {sparsity:.0%}: removed {rows[-1]['removed_percent']:.2f}%, "
              f"loss damage {rows[-1]['loss_damage_percent']:.2f}%", flush=True)
        del model
    save(args.output, {**common, "status": "complete",
                       "stop_index": stop_index, "curve": rows})
