#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning.paths import ROOT
from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain
from pruning.nnprune.factorized_fisher import complement_update, inverse_diagonal
from pruning.nnprune.fisher import PRESETS, resolve_strategy, validate_strategy

from pruning.baselines.result_artifacts import write_json
from pruning.experiments.opt_b_factors import estimate_b_covariance, inverse_factors
from pruning.experiments.opt_cpu import (calibration_tokens, linear_layers, per_sequence_loss,
                           selection_sequences)


class StopCapture(Exception):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--model-dtype", choices=("auto", "float32", "float64", "float16", "bfloat16"),
                        default="auto")
    parser.add_argument("--targets", default=".02,.05,.1")
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--b-sequences", type=int, default=8)
    parser.add_argument("--b-context", type=int, default=128)
    parser.add_argument("--b-tokens", type=int, default=256)
    parser.add_argument("--b-batch-size", type=int, default=1)
    parser.add_argument("--b-damp-ratio", type=float, default=1e-6)
    parser.add_argument("--b-refresh", choices=("frozen", "layer"), default="frozen")
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--modules", default="all")
    parser.add_argument("--selection-sequences", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument("--heldout-sequences", type=int, default=0)
    parser.add_argument("--heldout-seed", type=int, default=314159)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--fisher-preset", choices=PRESETS,
                        default="fisher_sparsegpt")
    parser.add_argument("--fisher-geometry", choices=("axb", "axi"))
    parser.add_argument("--fisher-deletion", choices=("obs",))
    parser.add_argument("--fisher-compensation", choices=("sparsegpt", "complement"))
    parser.add_argument("--fisher-mask-rule", choices=("sparsegpt",),
                        default="sparsegpt")
    parser.add_argument("--fisher-cg-rtol", type=float, default=1e-6)
    parser.add_argument("--fisher-cg-max-iterations", type=int, default=1000)
    parser.add_argument("--fisher-update-scope", choices=("layer",))
    parser.add_argument("--fisher-structure", choices=("post", "none"))
    parser.add_argument("--fisher-decision", choices=("validate", "proxy"))
    parser.add_argument("--fisher-mu", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "outputs/manual/fisher_sparsegpt.json")
    args = parser.parse_args(argv)
    args.targets = tuple(float(value) for value in args.targets.split(","))
    positive = (args.nsamples, args.seqlen, args.b_sequences, args.b_context,
                args.b_tokens, args.b_batch_size, args.blocksize, args.max_layers,
                args.selection_sequences, args.threads)
    if (min(positive) < 1 or args.b_sequences > args.nsamples or
            args.heldout_sequences < 0 or args.heldout_sequences == 1):
        parser.error("sample and block settings must be positive")
    if (not args.targets or tuple(sorted(args.targets)) != args.targets or
            any(not 0 < value <= 1 for value in args.targets) or
            (1. in args.targets and not args.progressive)):
        parser.error("targets must be increasing fractions in (0, 1]; 1 requires --progressive")
    if args.progressive and args.resume:
        parser.error("progressive runs cannot resume without a saved model checkpoint")
    if min(args.b_damp_ratio, args.percdamp, args.rho, args.n_sigma) < 0:
        parser.error("damping must be non-negative")
    if args.fisher_mu < 0:
        parser.error("Fisher update damping must be non-negative")
    if not 0 < args.fisher_cg_rtol < 1 or args.fisher_cg_max_iterations < 1:
        parser.error("invalid complement CG convergence settings")
    try:
        args.fisher_strategy = validate_strategy(resolve_strategy(
            args.fisher_preset, geometry=args.fisher_geometry,
            deletion=args.fisher_deletion, compensation=args.fisher_compensation,
            update_scope=args.fisher_update_scope,
            structure=args.fisher_structure, decision=args.fisher_decision), "opt")
    except ValueError as error:
        parser.error(str(error))
    if args.b_refresh == "layer" and (
            args.fisher_strategy.geometry != "axb" or args.b_batch_size != 1):
        parser.error("per-layer B refresh requires AXB geometry and B batch size 1")
    if args.fisher_strategy.compensation == "complement" and not args.progressive:
        parser.error("full-complement compensation requires progressive pruning")
    if args.progressive and (args.fisher_strategy.compensation != "complement" or
                             args.fisher_mask_rule != "sparsegpt" or
                             args.fisher_strategy.decision != "validate"):
        parser.error("progressive OPT pruning requires validated full-complement updates with the SparseGPT mask rule")
    return args


@torch.inference_mode()
def batched_sequence_loss(model, sequences, batch_size=2):
    device = next(model.parameters()).device
    batches = DataLoader(TensorDataset(torch.cat(sequences)), batch_size=batch_size)
    losses = []
    for (ids,) in batches:
        ids = ids.to(device)
        logits = model(ids, use_cache=False).logits[:, :-1].float()
        token_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
            reduction="none")
        losses.append(token_loss.reshape(len(ids), -1).mean(1).cpu())
    return torch.cat(losses).double()


def selected_names(layer, selection):
    names = tuple(linear_layers(layer))
    if selection == "all":
        return names
    requested = tuple(value.strip() for value in selection.split(",") if value.strip())
    missing = set(requested) - set(names)
    if missing:
        raise ValueError(f"unknown linear modules: {sorted(missing)}")
    return requested


@torch.inference_mode()
def capture_first_inputs(model, sequences):
    layers = model.model.decoder.layers
    original = layers[0]
    hidden, kwargs = [], []
    device = next(model.parameters()).device

    class Catcher(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, values, **arguments):
            hidden.append(values.detach().cpu())
            kwargs.append({key: (value.detach().cpu() if torch.is_tensor(value) else value)
                           for key, value in arguments.items()})
            raise StopCapture

    layers[0] = Catcher(original)
    try:
        for sequence in sequences:
            try:
                model(sequence.to(device), use_cache=False)
            except StopCapture:
                pass
    finally:
        layers[0] = original
    return torch.cat(hidden), kwargs


def forward_layer(layer, hidden, kwargs):
    device = next(layer.parameters()).device
    output = torch.empty_like(hidden)
    with torch.inference_mode():
        for index in range(len(hidden)):
            arguments = {key: (value.to(device) if torch.is_tensor(value) else value)
                         for key, value in kwargs[index].items()}
            result = layer(hidden[index:index + 1].to(device), **arguments)[0]
            output[index:index + 1] = result.cpu()
    return output


class SecondMoment:
    def __init__(self, width):
        self.matrix = torch.zeros((width, width))
        self.count = 0

    def add(self, values):
        values = values.detach().reshape(-1, values.shape[-1]).float().cpu()
        self.matrix.addmm_(values.T, values)
        self.count += len(values)

    def value(self):
        return self.matrix / self.count


def collect_a(layer, hidden, kwargs, names):
    modules = linear_layers(layer)
    stats = {name: SecondMoment(modules[name].in_features) for name in names}
    handles = [modules[name].register_forward_pre_hook(
        lambda _module, inputs, key=name: stats[key].add(inputs[0])) for name in names]
    try:
        forward_layer(layer, hidden, kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return {name: value.value() for name, value in stats.items()}


def estimate_b(model, sequences, layers_count, names, max_tokens, batch_size, seed):
    modules = {(index, name): linear_layers(layer)[name]
               for index, layer in enumerate(model.model.decoder.layers[:layers_count])
               for name in names}
    samples = {key: [] for key in modules}
    counts = {key: 0 for key in modules}
    current, handles = {}, []
    for key, module in modules.items():
        def capture(_module, _inputs, output, name=key):
            current[name] = output
            output.retain_grad()
        handles.append(module.register_forward_hook(capture))
    required = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embed_handle = model.model.decoder.embed_tokens.register_forward_hook(
        lambda _module, _inputs, output: output.detach().requires_grad_(True))
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batches = list(DataLoader(TensorDataset(torch.cat(sequences)), batch_size=batch_size))
    quota = max(1, math.ceil(max_tokens / len(batches)))
    started = time.perf_counter()
    try:
        with torch.enable_grad():
            for (ids,) in batches:
                if min(counts.values()) >= max_tokens:
                    break
                current.clear()
                ids = ids.to(device)
                logits = model(ids, use_cache=False).logits[:, :-1].float()
                probabilities = torch.softmax(logits.detach(), -1)
                labels = torch.multinomial(
                    probabilities.reshape(-1, probabilities.shape[-1]), 1,
                    generator=generator).reshape(logits.shape[:-1])
                torch.log_softmax(logits, -1).gather(
                    -1, labels.unsqueeze(-1)).mean().backward()
                for key, output in current.items():
                    gradient = output.grad.reshape(*ids.shape, -1)[:, :-1].reshape(
                        -1, output.shape[-1]).float()
                    gradient.mul_(labels.numel())
                    count = min(quota, max_tokens - counts[key], len(gradient))
                    positions = torch.linspace(
                        0, len(gradient) - 1, count, device=device).long()
                    samples[key].append(gradient[positions].detach().cpu())
                    counts[key] += count
    finally:
        embed_handle.remove()
        for handle in handles:
            handle.remove()
        for parameter, value in zip(model.parameters(), required):
            parameter.requires_grad_(value)
    return {key: torch.cat(value) for key, value in samples.items()}, {
        "seconds": time.perf_counter() - started,
        "observations": min(counts.values()),
    }


def downstream_factors(samples, damping_ratio, preserve_precision=False):
    factors, inverse_diagonals, off_diagonal = {}, {}, []
    for key, gradients in samples.items():
        gradients = gradients.double()
        factor = gradients.T @ gradients / len(gradients)
        diagonal = torch.diag(factor.diag())
        off_diagonal.append(float((factor - diagonal).norm() /
                                  factor.norm().clamp_min(1e-30)))
        damping = damping_ratio * max(float(factor.diag().mean()),
                                      torch.finfo(torch.float64).eps)
        factor.diagonal().add_(damping)
        factors[key] = factor if preserve_precision else factor.float()
        if damping:
            scaled = gradients / math.sqrt(len(gradients))
            middle = (torch.eye(len(gradients), dtype=torch.float64) +
                      scaled @ scaled.T / damping)
            solved = torch.linalg.solve(middle, scaled)
            inverse_diagonals[key] = (torch.full(
                (gradients.shape[1],), 1 / damping, dtype=torch.float64) -
                (scaled * solved).sum(0) / damping ** 2).clamp_min(1e-30)
        else:
            inverse_diagonals[key] = inverse_diagonal(factor)
    return factors, inverse_diagonals, {
        "minimum": min(off_diagonal), "mean": sum(off_diagonal) / len(off_diagonal),
        "maximum": max(off_diagonal),
    }


@torch.inference_mode()
def fisher_sparsegpt(module, a, b, b_inverse_diagonal, sparsity,
                     percdamp, blocksize):
    weight = module.weight.detach().float().cpu().clone()
    a = ((a + a.T) / 2).clone()
    dead = a.diag() == 0
    a[dead, dead] = 1
    weight[:, dead] = 0
    diagonal = torch.arange(len(a))
    a[diagonal, diagonal] += percdamp * a.diag().mean()
    factor, info = torch.linalg.cholesky_ex(a)
    if int(info):
        raise RuntimeError("damped activation factor is not positive definite")
    inverse_factor = torch.linalg.cholesky(
        torch.cholesky_inverse(factor), upper=True)
    predicted = 0.
    for start in range(0, weight.shape[1], blocksize):
        end = min(start + blocksize, weight.shape[1])
        current = weight[:, start:end].clone()
        result = torch.zeros_like(current)
        errors = torch.zeros_like(current)
        local_inverse = inverse_factor[start:end, start:end]
        input_inverse = local_inverse.diag().double().square()
        denominator = (input_inverse[None, :] if b_inverse_diagonal is None else
                       b_inverse_diagonal[:, None] * input_inverse[None, :])
        denominator = denominator.clamp_min(1e-30)
        score = current.double().square() / denominator
        count = min(int(score.numel() * sparsity), score.numel() - 1)
        threshold = torch.kthvalue(score.flatten(), count + 1).values
        mask = score <= threshold
        for column in range(end - start):
            values = current[:, column]
            scale = local_inverse[column, column]
            quantized = values.clone()
            quantized[mask[:, column]] = 0
            result[:, column] = quantized
            error = (values - quantized) / scale
            current[:, column:] -= error[:, None] @ local_inverse[
                column, column:][None, :]
            errors[:, column] = error
        fisher_errors = errors if b is None else errors.to(b.dtype)
        predicted += .5 * float(fisher_errors.square().sum() if b is None else
                                (fisher_errors * (b @ fisher_errors)).sum())
        weight[:, start:end] = result
        weight[:, end:] -= errors @ inverse_factor[start:end, end:]
    module.weight.copy_(weight.to(module.weight))
    return {"deleted": int((weight == 0).sum()),
            "predicted_fisher_damage": predicted}


@torch.inference_mode()
def fisher_sparsegpt_mask(module, a, b_inverse_diagonal, sparsity, percdamp, blocksize):
    original = module.weight.detach().clone()
    try:
        fisher_sparsegpt(module, a, None, b_inverse_diagonal, sparsity,
                         percdamp, blocksize)
        return (module.weight == 0).detach().clone()
    finally:
        module.weight.copy_(original)


@torch.inference_mode()
def fisher_layer_complement(module, a, b, deletion_mask, args):
    started = time.perf_counter()
    weight = module.weight.detach().double().clone()
    a = a.to(weight)
    a = ((a + a.T) / 2).clone()
    a.diagonal().add_(args.percdamp * a.diag().mean())
    b = b.to(weight)
    original_zeros = weight == 0
    if deletion_mask.shape != weight.shape or deletion_mask.dtype != torch.bool:
        raise ValueError("deletion mask must match the module weight matrix")
    deleted = deletion_mask.to(weight.device).clone() | original_zeros
    count = int((deleted & ~original_zeros).sum())
    delta, record = complement_update(
        weight, a, b, deleted, mu=args.fisher_mu,
        rtol=args.fisher_cg_rtol, max_iterations=args.fisher_cg_max_iterations,
        progress=lambda iteration, residual: print(
            f"Complement {tuple(weight.shape)}: CG {iteration}, "
            f"relative residual {residual:.3g}, {time.perf_counter() - started:.1f}s",
            flush=True))
    result = (weight + delta).masked_fill(deleted, 0.)
    module.weight.copy_(result.to(module.weight))
    return {**record, "deleted": count, "actual_zeros": int(deleted.sum()),
            "updates": int(torch.count_nonzero(delta.masked_fill(deleted, 0.))),
            "seconds": time.perf_counter() - started,
            "deletion_selection": "Fisher-aware SparseGPT blockwise mask; temporary updates discarded"}


def parameter_count(model, layers_count, names):
    return sum(module.weight.numel() +
               (module.bias.numel() if module.bias is not None else 0)
               for layer in model.model.decoder.layers[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def nonzero_count(model, layers_count, names):
    return sum(int(torch.count_nonzero(module.weight)) +
               (int(torch.count_nonzero(module.bias)) if module.bias is not None else 0)
               for layer in model.model.decoder.layers[:layers_count]
               for name, module in linear_layers(layer).items() if name in names)


def peak_memory_mb(device):
    return {"process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_allocated": (torch.cuda.max_memory_allocated(device) / 2 ** 20
                               if device.type == "cuda" else 0.)}


def prune_model(model, calibration, target, b_factors, b_inverse, layers_count,
                names, args, progress, b_calibration=None):
    started = time.perf_counter()
    hidden, kwargs = capture_first_inputs(model, calibration)
    history, canonical_savings = [], 0
    for layer_index, layer in enumerate(model.model.decoder.layers[:layers_count]):
        layer_started = time.perf_counter()
        b_seconds = 0.
        b_observations = 0
        layer_b_factors, layer_b_inverse = b_factors, b_inverse
        if args.b_refresh == "layer":
            covariance, counts, b_seconds = estimate_b_covariance(
                model, b_calibration, names, args.b_tokens, args.seed,
                layers_count, layer_index)
            factors = inverse_factors(covariance, counts, args.b_damp_ratio)
            layer_b_factors = {key: value[0] for key, value in factors.items()}
            layer_b_inverse = {key: value[1].diag() for key, value in factors.items()}
            b_observations = min(counts.values())
        a_factors = collect_a(layer, hidden, kwargs, names)
        modules = linear_layers(layer)
        records = {}
        for name in names:
            key = (layer_index, name)
            if args.fisher_strategy.compensation == "complement":
                mask = fisher_sparsegpt_mask(
                    modules[name], a_factors[name], layer_b_inverse[key], target,
                    args.percdamp, args.blocksize)
                records[name] = fisher_layer_complement(
                    modules[name], a_factors[name], layer_b_factors[key], mask, args)
            else:
                records[name] = fisher_sparsegpt(
                    modules[name], a_factors[name],
                    None if layer_b_factors is None else layer_b_factors[key],
                    None if layer_b_inverse is None else layer_b_inverse[key],
                    target, args.percdamp, args.blocksize)
        before = nonzero_count(model, layer_index + 1, names)
        if (args.fisher_strategy.structure == "post" and
                "fc1" in names and "fc2" in names):
            canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))
        after = nonzero_count(model, layer_index + 1, names)
        canonical_savings += max(0, before - after)
        hidden = forward_layer(layer, hidden, kwargs)
        history.append({"layer": layer_index, "modules": records,
                        "b_estimation_seconds": b_seconds,
                        "b_observations": b_observations,
                        "canonical_savings": canonical_savings,
                        "seconds": time.perf_counter() - layer_started})
        progress(layer_index + 1, history, canonical_savings,
                 time.perf_counter() - started)
        print(f"target {target:.0%}, layer {layer_index + 1}/{layers_count}: "
              f"{history[-1]['seconds']:.1f}s", flush=True)
    return history, canonical_savings, time.perf_counter() - started


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print("Loading calibration sequences", flush=True)
    calibration = calibration_tokens(tokenizer, "c4", args.nsamples, args.seqlen,
                                     args.seed)
    diagnostic = selection_sequences(tokenizer, "c4", args.seqlen, 0,
                                     args.selection_sequences, args.selection_seed)
    heldout = (selection_sequences(tokenizer, "c4", args.seqlen, 0,
                                   args.heldout_sequences, args.heldout_seed)
               if args.heldout_sequences else None)
    print("Loading reference model", flush=True)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=args.model_dtype).to(device).eval()
    reference_model.config.use_cache = False
    layers_count = min(args.max_layers, len(reference_model.model.decoder.layers))
    names = selected_names(reference_model.model.decoder.layers[0], args.modules)
    reference_loss = per_sequence_loss(reference_model, diagnostic).double()
    heldout_reference = (batched_sequence_loss(reference_model, heldout)
                         if heldout is not None else None)
    print(f"Reference loss: {float(reference_loss.mean()):.6f}", flush=True)
    baseline = float(reference_loss.mean())
    denominator = parameter_count(reference_model, layers_count, names)
    initial_nonzero = nonzero_count(reference_model, layers_count, names)
    b_sequences = None
    if args.fisher_strategy.geometry == "axb":
        b_sequences = [value[:, :min(args.seqlen, args.b_context)]
                       for value in calibration[:args.b_sequences]]
        if args.b_refresh == "frozen":
            b_samples, b_timing = estimate_b(
                reference_model, b_sequences, layers_count, names, args.b_tokens,
                args.b_batch_size, args.seed)
            print(f"B gradients collected in {b_timing['seconds']:.1f}s", flush=True)
            b_factors, b_inverse, off_diagonal = downstream_factors(
                b_samples, args.b_damp_ratio,
                preserve_precision=args.fisher_strategy.compensation == "complement")
            del b_samples
        else:
            b_factors = b_inverse = None
            b_timing = {"seconds": 0., "observations": 0}
            off_diagonal = None
    else:
        b_factors = b_inverse = None
        b_timing = {"seconds": 0., "observations": 0}
        off_diagonal = None
    del reference_model

    strategy = args.fisher_strategy.as_dict()
    b_description = ("identity" if args.fisher_strategy.geometry == "axi" else
                     "refreshed at each transformer block" if args.b_refresh == "layer" else
                     "dense, estimated once from the original model")
    approximations = [
        ("layerwise K-FAC uses F_W approximately equal to A kron B"
         if args.fisher_strategy.geometry == "axb" else
         "A-only geometry uses F_W approximately equal to A kron I")]
    if args.fisher_strategy.geometry == "axb":
        approximations += [
            ("B is refreshed from the current model before each transformer block"
             if args.b_refresh == "layer" else
             "B is frozen at the original model"),
            "sampled token score gradients omit most cross-token covariance",
        ]
    complement = args.fisher_strategy.compensation == "complement"
    if complement:
        approximations += [
            "deletions use the Fisher-aware SparseGPT blockwise adaptive mask; its temporary updates are discarded",
            "all surviving module weights compensate via a matrix-free constrained quadratic solve",
            "A/B stay fixed within each module; biases remain fixed",
            "accepted sparsity steps reuse the current pruned model; each step refreshes A/B and updates all surviving module weights",
        ]
    else:
        approximations += [
            "the block mask uses scalar generalized-OBS scores and ignores interactions among simultaneous deletions",
            "SparseGPT fixes each processed column, so compensation is unrestricted only in later input columns",
        ]

    result = {
        "result_schema": 3, "status": "running",
        "method": ("Fisher progressive full-layer complement (SparseGPT mask)" if complement else
                   "SparseGPT (A-only)" if args.fisher_strategy.geometry == "axi"
                   else "Fisher-aware SparseGPT"),
        "model": args.model, "model_dtype": args.model_dtype,
        "fisher_strategy": strategy,
        "calibration": {"dataset": "c4", "nsamples": args.nsamples,
                        "seqlen": args.seqlen, "seed": args.seed,
                        "b_sequences": args.b_sequences,
                        "b_context": args.b_context, "b_tokens": args.b_tokens},
        "protocol": {"blocksize": args.blocksize, "percdamp": args.percdamp,
                     "progressive": args.progressive,
                     "mask_rule": args.fisher_mask_rule,
                     "b_damp_ratio": args.b_damp_ratio,
                     "b_refresh": args.b_refresh,
                     "B": b_description,
                     "A": "collected once per layer from the current layer inputs",
                     "mask": ("Fisher-aware SparseGPT blockwise adaptive mask" if complement else
                              "generalized OBS saliency"),
                     "compensation": ("full-complement constrained AXB solve" if complement else
                                      "SparseGPT columnwise error propagation"),
                     "fisher_mu": args.fisher_mu,
                     "fisher_cg_rtol": args.fisher_cg_rtol,
                     "fisher_cg_max_iterations": args.fisher_cg_max_iterations},
        "approximations": approximations,
        "b_estimation": {**b_timing, "mode": args.b_refresh,
                         "off_diagonal_frobenius_ratio": off_diagonal},
        "parameter_denominator": denominator,
        "selection": {"dataset": "c4", "seed": args.selection_seed,
                      "count": args.selection_sequences, "rho": args.rho,
                      "n_sigma": args.n_sigma},
        "heldout": ({"dataset": "c4", "seed": args.heldout_seed,
                     "count": args.heldout_sequences,
                     "reference_loss": float(heldout_reference.mean())}
                    if heldout is not None else None),
        "curve": [{"target_sparsity": 0., "removed_percent": 0.,
                   "loss_damage_percent": 0., "selection_loss": baseline}],
        "runs": [],
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("fisher_strategy") != strategy:
            raise ValueError("resume artifact uses a different Fisher strategy")
        completed = [run for run in previous.get("runs", [])
                     if run.get("status") == "complete"]
        completed_targets = {run["target_sparsity"] for run in completed}
        result["runs"] = completed
        result["curve"].extend(
            row for row in previous.get("curve", [])[1:]
            if row.get("target_sparsity") in completed_targets)
    else:
        completed_targets = set()
    write_json(args.output, result)
    model = None
    for target in args.targets:
        if target in completed_targets:
            continue
        print(f"Starting target {target:.0%}", flush=True)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=args.model_dtype).to(device).eval()
        model.config.use_cache = False
        run = {"target_sparsity": target, "status": "running", "layers": [],
               "starting_removed_percent": 100 * (
                   initial_nonzero - nonzero_count(model, layers_count, names)) / denominator}
        result["runs"].append(run)

        def progress(completed, history, canonical_savings, seconds):
            run.update({"completed_layers": completed, "layers": history,
                        "canonical_savings": canonical_savings,
                        "seconds": seconds})
            write_json(args.output, result)

        try:
            history, canonical_savings, seconds = prune_model(
                model, calibration, target, b_factors, b_inverse, layers_count,
                names, args, progress, b_calibration=b_sequences)
        except Exception as error:
            run.update(status="failed", error=str(error))
            result.update(status="failed", termination_reason="solver_or_pruning_error",
                          error=str(error))
            write_json(args.output, result)
            raise
        remaining = nonzero_count(model, layers_count, names)
        values = per_sequence_loss(model, diagnostic).double()
        difference = values - reference_loss
        row = {
            "target_sparsity": target,
            "removed_percent": 100 * (initial_nonzero - remaining) / denominator,
            "loss_damage_percent": 100 * (float(values.mean()) / baseline - 1),
            "selection_loss": float(values.mean()),
            "selection_delta": float(difference.mean()),
            "selection_se": float(difference.std(unbiased=True) / math.sqrt(len(difference))),
            "canonical_savings": canonical_savings, "seconds": seconds,
            "b_estimation_seconds": sum(layer["b_estimation_seconds"] for layer in history),
        }
        if heldout is not None:
            heldout_values = batched_sequence_loss(model, heldout)
            heldout_difference = heldout_values - heldout_reference
            row.update({
                "heldout_loss": float(heldout_values.mean()),
                "heldout_damage_percent": 100 * (
                    float(heldout_values.mean() / heldout_reference.mean()) - 1),
                "heldout_delta": float(heldout_difference.mean()),
                "heldout_se": float(heldout_difference.std(unbiased=True) /
                                    math.sqrt(len(heldout_difference))),
            })
        row["admissible"] = (row["selection_delta"] <=
                             args.n_sigma * row["selection_se"] +
                             args.rho * baseline)
        result["curve"].append(row)
        run.update({"status": "complete", "layers": history, **row,
                    "peak_memory_mb": peak_memory_mb(device)})
        if args.save_model and (target == args.targets[-1] or not row["admissible"]):
            model_path = args.output.with_name("model.pt")
            torch.save({"state_dict": model.state_dict(),
                        "target_sparsity": target,
                        "removed_percent": row["removed_percent"]}, model_path)
            result["model_artifact"] = model_path.name
        write_json(args.output, result)
        if not args.progressive:
            del model
            model = None
        if args.fisher_strategy.decision == "validate" and not row["admissible"]:
            result["termination_reason"] = "loss_limit"
            break
    else:
        result["termination_reason"] = "targets_exhausted"
    result["accepted_endpoint"] = next(
        row for row in reversed(result["curve"])
        if row.get("admissible", True))
    result["status"] = "complete"
    write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
