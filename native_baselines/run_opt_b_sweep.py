#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from result_artifacts import write_json
from run_opt_fisher_sparsegpt import (capture_first_inputs, collect_a,
                                      fisher_layer_complement,
                                      fisher_layer_miqp_repeated, fisher_sparsegpt,
                                      fisher_sparsegpt_axb,
                                      forward_layer, linear_layers,
                                      parameter_count, selected_names)
from run_cpu_close import calibration_tokens, selection_sequences
from nnprune.fisher import resolve_strategy


CASES = (
    ("sparsegpt_b256", "sparsegpt", 8, 128, 256, 1e-6, "frozen"),
    ("axb_b256_d1e6", "axb", 8, 128, 256, 1e-6, "frozen"),
    ("axb_b256_d1e2", "axb", 8, 128, 256, 1e-2, "frozen"),
    ("axb_b1016_d1e6", "axb", 8, 128, 1016, 1e-6, "frozen"),
    ("axb_b4064_d1e6", "axb", 32, 128, 4064, 1e-6, "frozen"),
    ("axb_b16256_d1e6", "axb", 128, 128, 16256, 1e-6, "frozen"),
    ("axb_b16256_d1e2", "axb", 128, 128, 16256, 1e-2, "frozen"),
    ("axb_b16256_d1e2_refresh_layer", "axb", 128, 128, 16256, 1e-2, "layer"),
    ("axb_b262016_d1e6", "axb", 128, 2048, 262016, 1e-6, "frozen"),
    ("axb_b262016_d1e2", "axb", 128, 2048, 262016, 1e-2, "frozen"),
)
MIQP_CASE = ("miqp_dense_b16256_d1e2", "miqp_dense", 128, 128, 16256, 1e-2, "frozen")
COMPLEMENT_CASE = ("axb_complement_b16256_d1e2_refresh_layer", "complement",
                   128, 128, 16256, 1e-2, "layer")


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


def sequence_losses(model, sequences):
    device = next(model.parameters()).device
    result = []
    for (ids,) in DataLoader(TensorDataset(torch.cat(sequences)), batch_size=1):
        with torch.inference_mode():
            ids = ids.to(device)
            logits = model(ids, use_cache=False).logits[:, :-1].float()
            result.append(float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), ids[:, 1:].flatten())))
    return result


def evaluate(model, splits, reference):
    result = {}
    for name, sequences in splits.items():
        values = sequence_losses(model, sequences)
        original = reference[name]
        result[name] = {
            "loss": sum(values) / len(values),
            "damage_percent": 100 * (sum(values) / sum(original) - 1),
            "paired_deltas": [value - base for value, base in zip(values, original)],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--heldout-count", type=int, default=8)
    parser.add_argument("--heldout-seed", type=int, default=54321)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent /
                        "results/opt_published/b_estimation_sweep_10pct.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    selected_cases = CASES if args.cases == "all" else tuple(
        case for case in CASES + (MIQP_CASE, COMPLEMENT_CASE)
        if case[0] in args.cases.split(","))
    if not selected_cases:
        parser.error("no cases selected")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    calibration = calibration_tokens(tokenizer, "c4", 128, 2048, 0)
    splits = {
        "selection": selection_sequences(tokenizer, "c4", 2048, 0, 4, 12345),
        "heldout": selection_sequences(
            tokenizer, "c4", 2048, 0, args.heldout_count, args.heldout_seed),
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="float32").to(device).eval()
    model.config.use_cache = False
    layers_count = min(args.max_layers, len(model.model.decoder.layers))
    names = selected_names(model.model.decoder.layers[0], "all")
    denominator = parameter_count(model, layers_count, names)
    miqp_args = SimpleNamespace(
        percdamp=.01, fisher_pool=128, fisher_mu=0.,
        fisher_lambda_u=0., fisher_lambda_d=0.,
        fisher_bound=10., fisher_time_limit=10., fisher_gap=.05,
        fisher_log=False,
        fisher_strategy=resolve_strategy(
            "fisher_sparsegpt", geometry="axb", deletion="miqp",
            compensation="dense", update_scope="layer",
            structure="none", decision="proxy"))
    complement_args = SimpleNamespace(
        percdamp=.01, fisher_mu=1e-4, fisher_cg_rtol=1e-6,
        fisher_cg_max_iterations=10000)
    reference = {name: sequence_losses(model, values) for name, values in splits.items()}
    del model
    result = {
        "model": args.model, "target_sparsity": .10,
        "calibration": {"dataset": "c4", "sequences": 128, "context": 2048, "seed": 0},
        "evaluation": {"selection_count": 4, "selection_seed": 12345,
                       "heldout_count": args.heldout_count,
                       "heldout_seed": args.heldout_seed,
                       "reference_losses": reference},
        "protocol": {"blocksize": 128, "a_damping": .01,
                     "structure": "none", "B": "sampled score-gradient covariance"},
        "parameter_denominator": denominator, "runs": [],
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("model") != args.model or previous.get("target_sparsity") != .10:
            raise ValueError("existing output has incompatible model or target")
        if (previous["evaluation"]["heldout_count"] != args.heldout_count or
                previous["evaluation"]["heldout_seed"] != args.heldout_seed):
            raise ValueError("existing output has incompatible heldout evaluation")
        result["runs"] = previous["runs"]
    for label, method, b_sequences, b_context, b_tokens, b_damping, refresh in selected_cases:
        if any(run["case"] == label and run["status"] == "complete"
               for run in result["runs"]):
            continue
        checkpoint = (args.output.with_suffix(".checkpoint.pt")
                      if method in ("miqp_dense", "complement") else None)
        resume = (checkpoint is not None and checkpoint.exists() and
                  any(run["case"] == label and run["status"] != "complete"
                      for run in result["runs"]))
        result["runs"] = [run for run in result["runs"] if run["case"] != label]
        run = {"case": label, "method": method, "b_sequences": b_sequences,
               "b_context": b_context, "b_tokens": b_tokens,
               "b_damping_ratio": b_damping, "b_refresh": refresh,
               "status": "running", "layers": []}
        if method == "miqp_dense":
            run["miqp"] = {
                "candidate_deletions": miqp_args.fisher_pool,
                "candidate_updates": miqp_args.fisher_pool,
                "deletions_per_move": miqp_args.fisher_pool // 2,
                "mu": miqp_args.fisher_mu,
                "lambda_u": miqp_args.fisher_lambda_u,
                "lambda_d": miqp_args.fisher_lambda_d,
                "reference_bound_multiplier": miqp_args.fisher_bound,
                "time_limit": miqp_args.fisher_time_limit,
                "gap": miqp_args.fisher_gap,
                "update_rule": "unconstrained continuous AXB optimum via Schur complement",
            }
        elif method == "complement":
            run["complement"] = {
                "deletion_rule": "generalized OBS ranking using AXB",
                "update_rule": "all surviving module weights via constrained AXB quadratic solve",
                "mu": complement_args.fisher_mu,
                "cg_rtol": complement_args.fisher_cg_rtol,
                "cg_max_iterations": complement_args.fisher_cg_max_iterations,
            }
        result["runs"].append(run)
        write_json(args.output, result)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="float32").to(device).eval()
        model.config.use_cache = False
        b_sequences_data = [value[:, :b_context] for value in calibration[:b_sequences]]
        hidden, kwargs = capture_first_inputs(model, calibration)
        factors = None
        removed = 0
        try:
            if refresh == "frozen":
                covariance, counts, seconds = estimate_b_covariance(
                    model, b_sequences_data, names, b_tokens, 0, layers_count)
                factors = inverse_factors(covariance, counts, b_damping)
                run["b_estimation_seconds"] = seconds
                run["b_observations"] = min(counts.values())
                write_json(args.output, result)
            starting_layer = 0
            if resume:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(saved["model"])
                run = saved["run"]
                result["runs"][-1] = run
                starting_layer = run["completed_layers"]
                removed = sum(record["deleted"]
                              for layer_record in run["layers"]
                              for record in layer_record["modules"].values())
                hidden, kwargs = capture_first_inputs(model, calibration)
                for layer in model.model.decoder.layers[:starting_layer]:
                    hidden = forward_layer(layer, hidden, kwargs)
            for layer_index in range(starting_layer, layers_count):
                layer = model.model.decoder.layers[layer_index]
                layer_started = time.perf_counter()
                if refresh == "layer":
                    covariance, counts, seconds = estimate_b_covariance(
                        model, b_sequences_data, names, b_tokens, 0,
                        layers_count, layer_index)
                    factors = inverse_factors(covariance, counts, b_damping)
                a_factors = collect_a(layer, hidden, kwargs, names)
                modules = linear_layers(layer)
                records = {}
                for name in names:
                    key = (layer_index, name)
                    module = modules[name]
                    before = module.weight.detach().clone()
                    b_factor, b_inverse = factors[key]
                    if method == "miqp_dense":
                        record = fisher_layer_miqp_repeated(
                            module, a_factors[name], b_factor, b_inverse.diag(),
                            .10, miqp_args)
                    elif method == "complement":
                        record = fisher_layer_complement(
                            module, a_factors[name], b_factor, b_inverse.diag(),
                            .10, complement_args)
                    elif method == "sparsegpt":
                        record = fisher_sparsegpt(
                            module, a_factors[name], None, b_inverse.diag(),
                            .10, .01, 128)
                    else:
                        record = fisher_sparsegpt_axb(
                            module, a_factors[name], None, b_inverse.diag(),
                            .10, .01, 128, b_damping, b_inverse, b_factor)
                    change = module.weight.detach() - before
                    surviving = module.weight.detach() != 0
                    record["survivor_change_norm"] = float(change[surviving].norm())
                    record["survivor_change_max"] = float(change[surviving].abs().max())
                    records[name] = record
                    removed += record["deleted"]
                    if refresh == "layer":
                        del factors[key]
                if refresh == "layer":
                    factors = None
                hidden = forward_layer(layer, hidden, kwargs)
                run["layers"].append({"layer": layer_index, "modules": records,
                                      "b_estimation_seconds": seconds if refresh == "layer" else None,
                                      "seconds": time.perf_counter() - layer_started})
                run["completed_layers"] = layer_index + 1
                if checkpoint is not None:
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save({"model": model.state_dict(), "run": run}, temporary)
                    temporary.replace(checkpoint)
                write_json(args.output, result)
                print(label, "layer", layer_index + 1, "of", layers_count,
                      "elapsed", round(time.perf_counter() - started, 1), flush=True)
            run.update(evaluate(model, splits, reference))
            run["removed_percent"] = 100 * removed / denominator
            run["seconds"] = time.perf_counter() - started
            run["status"] = "complete"
            write_json(args.output, result)
            if checkpoint is not None:
                checkpoint.unlink(missing_ok=True)
            print("RESULT", json.dumps({key: value for key, value in run.items()
                                        if key != "layers"}), flush=True)
        except Exception as error:
            run.update(status="failed", error=str(error),
                       seconds=time.perf_counter() - started)
            write_json(args.output, result)
            raise
        finally:
            del model, factors, hidden


if __name__ == "__main__":
    main()
