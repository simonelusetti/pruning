#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from result_artifacts import write_json
from run_cpu_close import calibration_tokens, selection_sequences
from run_opt_b_sweep import (estimate_b_covariance, evaluate, inverse_factors,
                             sequence_losses)
from run_opt_fisher_sparsegpt import (capture_first_inputs, collect_a,
                                      fisher_sparsegpt_axb, forward_layer,
                                      linear_layers, parameter_count,
                                      selected_names)
from nnprune.factorized_fisher import complement_update


def damped_a(a):
    a = ((a.float() + a.float().T) / 2).clone()
    dead = a.diag() == 0
    a[dead, dead] = 1
    a.diagonal().add_(.01 * a.diag().mean())
    return a.double(), dead


def quadratic(delta, a, b):
    return float(.5 * (delta * (b @ delta @ a)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--heldout-count", type=int, default=128)
    parser.add_argument("--mus", default="0,0.0001")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mus = tuple(float(value) for value in args.mus.split(","))
    if (min(args.threads, args.max_layers, args.heldout_count) < 1 or
            not mus or min(mus) < 0 or len(set(mus)) != len(mus)):
        parser.error("threads, layers, and held-out count must be positive; mus must be distinct and nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    calibration = calibration_tokens(tokenizer, "c4", 128, 2048, 0)
    b_sequences = [value[:, :128] for value in calibration]
    splits = {
        "selection": selection_sequences(tokenizer, "c4", 2048, 0, 4, 12345),
        "heldout": selection_sequences(tokenizer, "c4", 2048, 0, args.heldout_count, 314159),
    }
    sparsegpt = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", dtype="float32").to(device).eval()
    complement_models = {
        f"full_mu{mu:g}": AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m", dtype="float32").to(device).eval()
        for mu in mus
    }
    mu_by_arm = {f"full_mu{mu:g}": mu for mu in mus}
    sparsegpt.config.use_cache = False
    for model in complement_models.values():
        model.config.use_cache = False
    layers_count = min(args.max_layers, len(sparsegpt.model.decoder.layers))
    names = selected_names(sparsegpt.model.decoder.layers[0], "all")
    reference = {name: sequence_losses(sparsegpt, values)
                 for name, values in splits.items()}
    result = {
        "status": "running", "model": "facebook/opt-125m",
        "target_sparsity": .10, "completed_layers": 0,
        "protocol": {
            "calibration": "128 C4 sequences of 2048 tokens; seed 0",
            "B": "16,256 sampled score gradients from 128x128 contexts; damping 0.01",
            "factor_source": "SparseGPT model before each layer; identical A and B for both update arms",
            "D": "AXB-aware SparseGPT adaptive mask, 128-column blocks; identical in both arms",
            "sparsegpt_update": "B compensation in current column, A propagation to later columns",
            "complement_update": "all unmasked weights via constrained AXB solve",
            "a_damping": .01, "mus": mus, "cg_rtol": 1e-6,
            "inactive_inputs": "zeroed in both arms before quadratic evaluation",
            "canonicalization": "none", "biases": "fixed",
            "heldout_count": args.heldout_count, "heldout_seed": 314159,
            "selection_count": 4, "selection_seed": 12345,
        },
        "parameter_denominator": parameter_count(sparsegpt, layers_count, names),
        "reference_losses": reference, "layers": [],
    }
    checkpoint = args.output.with_suffix(".checkpoint.pt")
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["protocol"] != result["protocol"]:
            raise ValueError("checkpoint protocol does not match this run")
        sparsegpt.load_state_dict(saved["sparsegpt"])
        for arm, model in complement_models.items():
            model.load_state_dict(saved["complements"][arm])
        result["layers"] = saved["layers"]
        result["completed_layers"] = len(result["layers"])
    write_json(args.output, result)
    hidden, kwargs = capture_first_inputs(sparsegpt, calibration)
    for layer_index in range(result["completed_layers"]):
        hidden = forward_layer(sparsegpt.model.decoder.layers[layer_index], hidden, kwargs)
    try:
        for layer_index in range(result["completed_layers"], layers_count):
            layer_started = time.perf_counter()
            sparse_layer = sparsegpt.model.decoder.layers[layer_index]
            complement_layers = {
                arm: model.model.decoder.layers[layer_index]
                for arm, model in complement_models.items()}
            covariance, counts, b_seconds = estimate_b_covariance(
                sparsegpt, b_sequences, names, 16256, 0, layers_count, layer_index)
            b_factors = inverse_factors(covariance, counts, .01)
            a_factors = collect_a(sparse_layer, hidden, kwargs, names)
            modules = {}
            for name in names:
                sparse_module = linear_layers(sparse_layer)[name]
                before = sparse_module.weight.detach().double().clone()
                b, b_inverse = b_factors[(layer_index, name)]
                fisher_sparsegpt_axb(sparse_module, a_factors[name], None,
                                     b_inverse.diag(), .10, .01, 128, .01,
                                     b_inverse, b)
                mask = sparse_module.weight.detach() == 0
                a, dead = damped_a(a_factors[name])
                a = a.to(before)
                before_effective = before.masked_fill(dead.to(before.device)[None, :], 0.)
                b = b.to(before)
                sparse_delta = sparse_module.weight.detach().double() - before_effective
                updates = {"sparsegpt": {
                    "quadratic": quadratic(sparse_delta, a, b),
                    "update_norm": float(sparse_delta.masked_fill(mask, 0.).norm()),
                }}
                for arm, complement_layer in complement_layers.items():
                    complement_module = linear_layers(complement_layer)[name]
                    if not torch.equal(before, complement_module.weight.detach().double()):
                        raise RuntimeError(f"unmatched starting weights: layer {layer_index} {name} {arm}")
                    delta, diagnostics = complement_update(
                        before_effective, a, b, mask, mu=mu_by_arm[arm], rtol=1e-6,
                        max_iterations=10000)
                    with torch.no_grad():
                        complement_module.weight.copy_(
                            (before_effective + delta).masked_fill(mask, 0.).to(complement_module.weight))
                    complement_delta = complement_module.weight.detach().double() - before_effective
                    updates[arm] = {
                        "quadratic": quadratic(complement_delta, a, b),
                        "update_norm": float(complement_delta.masked_fill(mask, 0.).norm()),
                        "iterations": diagnostics["iterations"],
                        "relative_residual": diagnostics["relative_residual"],
                    }
                modules[name] = {
                    "deleted": int(mask.sum()),
                    "inactive_input_columns": int(dead.sum()),
                    "mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
                    "updates": updates,
                }
            hidden = forward_layer(sparse_layer, hidden, kwargs)
            result["layers"].append({"layer": layer_index, "modules": modules,
                                     "b_estimation_seconds": b_seconds,
                                     "seconds": time.perf_counter() - layer_started})
            result["completed_layers"] = layer_index + 1
            result["seconds"] = time.perf_counter() - started
            temporary = checkpoint.with_suffix(".tmp")
            torch.save({"protocol": result["protocol"], "layers": result["layers"],
                        "sparsegpt": sparsegpt.state_dict(),
                        "complements": {arm: model.state_dict()
                                        for arm, model in complement_models.items()}}, temporary)
            temporary.replace(checkpoint)
            write_json(args.output, result)
            print(f"matched updates layer {layer_index + 1}/{layers_count}: "
                  f"{result['layers'][-1]['seconds']:.1f}s", flush=True)
        result["arms"] = {"sparsegpt": evaluate(sparsegpt, splits, reference)}
        result["arms"].update({arm: evaluate(model, splits, reference)
                               for arm, model in complement_models.items()})
        result["removed_percent"] = 100 * sum(
            record["deleted"] for layer in result["layers"]
            for record in layer["modules"].values()) / result["parameter_denominator"]
        result["status"] = "complete"
        result["seconds"] = time.perf_counter() - started
        write_json(args.output, result)
        checkpoint.unlink(missing_ok=True)
        print("RESULT", json.dumps({
            "removed_percent": result["removed_percent"],
            "heldout_damage": {arm: value["heldout"]["damage_percent"]
                               for arm, value in result["arms"].items()},
            "seconds": result["seconds"]}), flush=True)
    except Exception as error:
        result.update(status="failed", error=str(error), seconds=time.perf_counter() - started)
        write_json(args.output, result)
        raise


if __name__ == "__main__":
    main()
