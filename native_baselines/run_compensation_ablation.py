#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))
from nnprune.factorized_fisher import complement_update, inverse_diagonal
from result_artifacts import write_json


ALPHAS = (0., .25, .5, .75, 1.)


def deletion_mask(weight, a, b, target):
    score = weight.square() / (
        inverse_diagonal(b).to(weight)[:, None] *
        inverse_diagonal(a).to(weight)[None, :]).clamp_min(1e-30)
    mask = weight == 0
    count = max(0, math.ceil(target * weight.numel()) - int(mask.sum()))
    score.masked_fill_(mask, torch.inf)
    mask.flatten()[torch.topk(score.flatten(), count, largest=False).indices] = True
    return mask


def quadratic_coefficients(fixed, update, action, mu):
    hf, hu = action(fixed), action(update)
    return {
        "constant": float(.5 * (fixed * hf).sum()),
        "linear": float((fixed * hu).sum()),
        "quadratic": float(.5 * (update * hu).sum()),
        "ridge": float(.5 * mu * update.square().sum()),
    }


def predicted(coefficients, alpha):
    q = coefficients["constant"] + alpha * coefficients["linear"] + alpha ** 2 * coefficients["quadratic"]
    return q, q + alpha ** 2 * coefficients["ridge"]


def check_solution(coefficients):
    baseline = coefficients["constant"]
    _, optimized = predicted(coefficients, 1.)
    if optimized > baseline + 1e-8 * max(abs(baseline), 1e-12):
        raise RuntimeError("compensation increased its own regularized quadratic")


@torch.inference_mode()
def axb_plan(module, a, b, args):
    weight = module.weight.detach().double().clone()
    a = ((a.to(weight) + a.to(weight).T) / 2).clone()
    a.diagonal().add_(.01 * a.diag().mean())
    b = b.to(weight)
    mask = deletion_mask(weight, a, b, args.target)
    delta, diagnostics = complement_update(
        weight, a, b, mask, mu=args.mu, rtol=1e-6,
        max_iterations=10000,
        progress=lambda iteration, residual: print(
            f"  {tuple(weight.shape)} CG {iteration}: {residual:.3g}", flush=True)
        if iteration % 250 == 0 else None)
    fixed = torch.where(mask, -weight, 0.)
    update = delta - fixed
    coefficients = quadratic_coefficients(fixed, update, lambda v: b @ v @ a, args.mu)
    check_solution(coefficients)
    digest = hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
    record = {**diagnostics, "coefficients": coefficients, "mask_sha256": digest,
              "deleted": int(mask.sum()), "weights": weight.numel(),
              "compensation_norm": float(update.norm())}
    # Every evaluated alpha uses these same tensors; no masks are recomputed.
    return (weight.to(module.weight), mask, update.to(module.weight)), record


@torch.inference_mode()
def apply_plan(modules, plans, alpha):
    for module, (weight, mask, update) in zip(modules, plans):
        module.weight.copy_((weight + alpha * update).masked_fill(mask, 0.))


def paired(values):
    values = values.double()
    return float(values.mean()), float(values.std(unbiased=True) / math.sqrt(len(values)))


@torch.inference_mode()
def kl_per_example(reference_logits, candidate_logits):
    ref = F.log_softmax(reference_logits.double(), -1)
    candidate = F.log_softmax(candidate_logits.double(), -1)
    return (ref.exp() * (ref - candidate)).sum(-1)


def exact_fisher_plan(plans, activations, gradients, args):
    shapes = [plan[0].shape for plan in plans]
    sizes = [weight.numel() for weight, _, _ in plans]
    mask = torch.cat([plan[1].flatten() for plan in plans])
    fixed = torch.cat([torch.where(d, -w.double(), 0.).flatten() for w, d, _ in plans])
    samples = len(activations[0])

    def action(vector):
        score = torch.zeros(gradients[0].shape[:2], dtype=vector.dtype, device=vector.device)
        matrices = vector.split(sizes)
        for flat, shape, a, g in zip(matrices, shapes, activations, gradients):
            perturbation = a[:, :-1] @ flat.reshape(shape).T
            score.add_(torch.einsum("ncr,nr->nc", g, perturbation))
        return torch.cat([
            (torch.einsum("ncr,nc->nr", g, score).T @ a[:, :-1] / samples).flatten()
            for a, g in zip(activations, gradients)])

    diagonal = torch.cat([
        (g.square().sum(1).T @ a[:, :-1].square() / samples).flatten()
        for a, g in zip(activations, gradients)]) + args.mu
    rhs = -action(fixed).masked_fill(mask, 0.)
    norm = rhs.norm()
    residual = rhs.clone()
    update = torch.zeros_like(rhs)
    direction = residual / diagonal
    rz = (residual * direction).sum()
    iteration = 0
    if float(norm):
        for iteration in range(1, 10001):
            product = (action(direction) + args.mu * direction).masked_fill(mask, 0.)
            curvature = (direction * product).sum()
            if float(curvature) <= 0 or not bool(torch.isfinite(curvature)):
                raise RuntimeError("exact Fisher CG lost positive definiteness")
            step = rz / curvature
            update.add_(step * direction)
            residual.sub_(step * product)
            if iteration % 100 == 0:
                print(f"Exact Fisher CG {iteration}: {float(residual.norm() / norm):.3g}", flush=True)
            if float(residual.norm()) <= 1e-6 * float(norm):
                residual = rhs - (action(update) + args.mu * update).masked_fill(mask, 0.)
                if float(residual.norm()) <= 1e-6 * float(norm):
                    break
                direction = residual / diagonal
                rz = (residual * direction).sum()
                continue
            z = residual / diagonal
            next_rz = (residual * z).sum()
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        if float(residual.norm()) > 1e-6 * float(norm):
            raise RuntimeError("exact Fisher CG failed residual check")
    coefficients = quadratic_coefficients(fixed, update, action, args.mu)
    check_solution(coefficients)
    exact_plans = [(w, d, flat.reshape(shape).to(w))
                   for (w, d, _), flat, shape in zip(plans, update.split(sizes), shapes)]
    return exact_plans, {"iterations": iteration, "relative_residual": float(residual.norm() / norm)
                         if float(norm) else 0., "coefficients": coefficients,
                         "scope": "all surviving weights across all layers; biases fixed"}


def run_berman(args, result):
    import run_berman_mnist as bm

    train, heldout = bm.mnist()
    order = torch.randperm(len(heldout[0]), generator=torch.Generator().manual_seed(bm.SPLIT_SEED))
    split_indices = {"validation": order[2500:5000], "test": order[5000:]}
    splits = {name: (heldout[0][indices], heldout[1][indices])
              for name, indices in split_indices.items()}
    splits["calibration"] = bm.fisher_images(train, 500)
    reference = bm.model(args.device)
    candidate = bm.model(args.device)
    images = bm.fisher_images(train, 500)[0].to(args.device)
    with torch.inference_mode():
        activations, gradients = bm.fisher_factors(reference, images)
        plans, records = [], []
        for module, a, g in zip(bm.linears(reference), activations, gradients):
            b = torch.einsum("nci,ncj->ij", g, g) / len(images)
            b.diagonal().add_(1e-6 * max(float(b.diag().mean()), torch.finfo(b.dtype).eps))
            plan, record = axb_plan(module, a[:, :-1].T @ a[:, :-1] / len(images), b, args)
            plans.append(plan)
            records.append(record)
    exact_plans, exact_record = exact_fisher_plan(plans, activations, gradients, args)
    result.update(module_records=records, exact_fisher=exact_record,
                  protocol={"checkpoint": str(bm.CHECKPOINT), "split_seed": bm.SPLIT_SEED,
                            "fisher_images": 500, "fisher_subset_seed": 42,
                            "biases": "fixed", "canonicalization": "none",
                            "geometry": "frozen at original model", "rho": .05, "n_sigma": 2.})
    modules = bm.linears(candidate)
    total = sum(p.numel() for p in reference.parameters())
    result["removed_percent"] = 100 * sum(int(p[1].sum()) for p in plans) / total
    result["evaluations"] = []
    loss_samples = {}
    for name, current_plans in (("axb", plans), ("exact_fisher", exact_plans)):
        for alpha in ALPHAS:
            apply_plan(modules, current_plans, alpha)
            entry = {"geometry": name, "alpha": alpha}
            for split, data in splits.items():
                reference_loss, candidate_loss, divergences = [], [], []
                correct, count = 0, 0
                for x, y in DataLoader(TensorDataset(*data), batch_size=256):
                    x, y = x.to(args.device), y.to(args.device)
                    with torch.inference_mode():
                        ref, out = reference(x), candidate(x)
                        reference_loss.append(F.cross_entropy(ref, y, reduction="none").cpu())
                        candidate_loss.append(F.cross_entropy(out, y, reduction="none").cpu())
                        divergences.append(kl_per_example(ref, out).cpu())
                        correct += int((out.argmax(-1) == y).sum())
                        count += len(y)
                ref_loss, out_loss = torch.cat(reference_loss), torch.cat(candidate_loss)
                loss_samples[(name, alpha, split)] = out_loss
                delta, se = paired(out_loss - ref_loss)
                entry[split] = {"reference_loss": float(ref_loss.mean()), "loss": float(out_loss.mean()),
                                "loss_damage_percent": 100 * (float(out_loss.mean() / ref_loss.mean()) - 1),
                                "paired_delta": delta, "paired_se": se,
                                "output_kl": float(torch.cat(divergences).mean()), "accuracy": correct / count}
                if split == "validation":
                    entry[split]["admissible"] = delta <= 2 * se + .05 * float(ref_loss.mean())
            coefficients = (exact_record["coefficients"] if name == "exact_fisher" else
                            {k: sum(r["coefficients"][k] for r in records) for k in records[0]["coefficients"]})
            entry["predicted_fisher_damage"], entry["regularized_objective"] = predicted(coefficients, alpha)
            result["evaluations"].append(entry)
            write_json(args.output, result)
            print(name, alpha, entry, flush=True)
    for entry in result["evaluations"]:
        for split in splits:
            samples = loss_samples[(entry["geometry"], entry["alpha"], split)]
            baseline = loss_samples[(entry["geometry"], 0., split)]
            delta, se = paired(samples - baseline)
            entry[split].update(compensation_delta=delta, compensation_se=se)


def run_opt(args, result):
    import run_opt_fisher_sparsegpt as opt
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    calibration = opt.calibration_tokens(tokenizer, "c4", 128, 2048, 0)
    # The selection split matches previous runs; the second split is independently seeded.
    splits = {"calibration": calibration[:4],
              "validation": opt.selection_sequences(tokenizer, "c4", 2048, 0, 4, 12345),
              "heldout": opt.selection_sequences(tokenizer, "c4", 2048, 0, 8, 54321)}
    reference = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", dtype="float32").to(args.device).eval()
    reference.config.use_cache = False
    names = opt.selected_names(reference.model.decoder.layers[0], "all")
    samples, timing = opt.estimate_b(reference, [x[:, :128] for x in calibration[:8]],
                                     12, names, 256, 1, 0)
    factors, _, off_diagonal = opt.downstream_factors(samples, 1e-6, preserve_precision=True)
    del samples
    hidden, kwargs = opt.capture_first_inputs(reference, calibration)
    plans, records = [], []
    for index, layer in enumerate(reference.model.decoder.layers):
        activations = opt.collect_a(layer, hidden, kwargs, names)
        for name, module in opt.linear_layers(layer).items():
            print(f"Building fixed-mask plan L{index} {name}", flush=True)
            plan, record = axb_plan(module, activations[name], factors[(index, name)], args)
            plans.append(tuple(v.cpu() for v in plan))
            records.append({"layer": index, "module": name, **record})
            result.update(module_records=records, completed_modules=len(records))
            write_json(args.output, result)
        hidden = opt.forward_layer(layer, hidden, kwargs)
    del factors, hidden, kwargs, activations
    candidate = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", dtype="float32").to(args.device).eval()
    candidate.config.use_cache = False
    modules = [m for layer in candidate.model.decoder.layers for m in opt.linear_layers(layer).values()]
    plans = [tuple(v.to(args.device) for v in plan) for plan in plans]
    coefficients = {k: sum(r["coefficients"][k] for r in records) for k in records[0]["coefficients"]}
    result.update(protocol={"model": "facebook/opt-125m", "dtype": "float32",
                            "calibration": "128x2048 C4; seed 0", "B": "8x128 contexts; 256 sampled scores",
                            "biases": "fixed", "canonicalization": "none",
                            "geometry": "original-model layerwise AXB; omits cross-layer terms",
                            "selection_seed": 12345, "heldout_seed": 54321,
                            "rho": .05, "n_sigma": 2.}, b_estimation=timing,
                  b_off_diagonal=off_diagonal, evaluations=[])
    denominator = opt.parameter_count(reference, 12, names)
    result["removed_percent"] = 100 * sum(int(p[1].sum()) for p in plans) / denominator
    measurements = {(split, alpha): {"reference": [], "loss": [], "kl": []}
                    for split in splits for alpha in ALPHAS}
    for split, sequences in splits.items():
        for (ids,) in DataLoader(TensorDataset(torch.cat(sequences)), batch_size=1):
            ids = ids.to(args.device)
            with torch.inference_mode():
                ref = reference(ids, use_cache=False).logits[:, :-1].float()
                labels = ids[:, 1:]
                ref_loss = float(F.cross_entropy(ref.reshape(-1, ref.shape[-1]), labels.flatten()))
                for alpha in ALPHAS:
                    apply_plan(modules, plans, alpha)
                    out = candidate(ids, use_cache=False).logits[:, :-1].float()
                    loss = float(F.cross_entropy(out.reshape(-1, out.shape[-1]), labels.flatten()))
                    divergence = 0.
                    for start in range(0, ref.shape[1], 128):
                        divergence += float(kl_per_example(ref[:, start:start + 128], out[:, start:start + 128]).sum())
                    measurement = measurements[(split, alpha)]
                    measurement["reference"].append(ref_loss)
                    measurement["loss"].append(loss)
                    measurement["kl"].append(divergence / labels.numel())
                    del out
            print(f"Evaluated {split} sequence", flush=True)
    for alpha in ALPHAS:
        entry = {"geometry": "axb", "alpha": alpha}
        for split in splits:
            measurement = measurements[(split, alpha)]
            ref_loss = torch.tensor(measurement["reference"], dtype=torch.float64)
            loss = torch.tensor(measurement["loss"], dtype=torch.float64)
            delta, se = paired(loss - ref_loss)
            entry[split] = {"reference_loss": float(ref_loss.mean()), "loss": float(loss.mean()),
                            "loss_damage_percent": 100 * (float(loss.mean() / ref_loss.mean()) - 1),
                            "paired_delta": delta, "paired_se": se,
                            "output_kl": sum(measurement["kl"]) / len(measurement["kl"])}
            baseline = torch.tensor(measurements[(split, 0.)]["loss"], dtype=torch.float64)
            compensation_delta, compensation_se = paired(loss - baseline)
            entry[split].update(compensation_delta=compensation_delta, compensation_se=compensation_se)
            if split == "validation":
                entry[split]["admissible"] = delta <= 2 * se + .05 * float(ref_loss.mean())
        entry["predicted_fisher_damage"], entry["regularized_objective"] = predicted(coefficients, alpha)
        result["evaluations"].append(entry)
        print(entry, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=("berman", "opt"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--target", type=float, default=.3)
    parser.add_argument("--mu", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.target < 1 or args.mu <= 0 or args.threads < 1:
        parser.error("target must be in (0,1), and mu and threads must be positive")
    args.device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    result = {"status": "running", "setting": args.setting, "target_sparsity": args.target,
              "mu": args.mu, "alphas": ALPHAS,
              "design": "fixed original-model AXB-OBS masks; interpolation changes compensation only"}
    started = time.perf_counter()
    write_json(args.output, result)
    try:
        (run_berman if args.setting == "berman" else run_opt)(args, result)
    except Exception as error:
        result.update(status="failed", error=str(error), seconds=time.perf_counter() - started)
        write_json(args.output, result)
        raise
    result.update(status="complete", seconds=time.perf_counter() - started)
    write_json(args.output, result)
    from plot_compensation_ablation import main as plot
    plot()


if __name__ == "__main__":
    main()
