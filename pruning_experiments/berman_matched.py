#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch

import pruning_experiments.berman as bm
from native_baselines.result_artifacts import write_json
from pruning_experiments.opt_fisher import fisher_sparsegpt_axb
from nnprune.factorized_fisher import complement_update


def damped_a(covariance):
    a = ((covariance.float() + covariance.float().T) / 2).clone()
    dead = a.diag() == 0
    a[dead, dead] = 1
    a.diagonal().add_(.01 * a.diag().mean())
    return a.double(), dead


def quadratic(delta, a, b):
    return float(.5 * (delta * (b @ delta @ a)).sum())


def evaluate(net, validation, test, reference, total, started, device):
    val_loss = bm.losses(net, validation, device)
    delta, se = bm.paired(val_loss, reference)
    removed = total - bm.nonzero(net)
    return {
        "removed": removed,
        "frac_removed": removed / total,
        "val_loss": float(val_loss.mean()),
        "val_delta": delta,
        "val_se": se,
        "test_loss": float(bm.losses(net, test, device).mean()),
        "test_acc": bm.accuracy(net, test, device),
        "seconds": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--targets", default=",".join(
        f"{step / 100:.2f}" for step in range(2, 72, 2)))
    parser.add_argument("--fisher-n", type=int, default=500)
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--output", type=Path, default=bm.RESULTS / "berman_matched_axb_updates.json")
    args = parser.parse_args(argv)
    targets = tuple(float(value) for value in args.targets.split(","))
    if (not targets or tuple(sorted(set(targets))) != targets or
            any(not 0 < target < 1 for target in targets) or
            min(args.threads, args.fisher_n) < 1 or min(args.rho, args.n_sigma) < 0):
        parser.error("invalid targets or experiment settings")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    train, heldout = bm.mnist()
    order = torch.randperm(len(heldout[0]), generator=torch.Generator().manual_seed(bm.SPLIT_SEED))
    subset = lambda indices: (heldout[0][order[indices]], heldout[1][order[indices]])
    validation = bm.loader(subset(slice(2500, 5000)), 512)
    test = bm.loader(subset(slice(5000, None)), 512)
    fisher_images = bm.fisher_images(train, args.fisher_n)[0].to(device)
    reference_model = bm.model(device)
    reference = bm.losses(reference_model, validation, device)
    base_loss = float(reference.mean())
    reference_test_loss = float(bm.losses(reference_model, test, device).mean())
    total = sum(parameter.numel() for parameter in reference_model.parameters())
    started = time.perf_counter()
    result = {
        **bm.COMMON,
        "status": "running",
        "method": "matched_axb_obs_deletions_full_complement",
        "protocol": {
            "split_seed": bm.SPLIT_SEED,
            "val_a": 2500,
            "val_b": 2500,
            "test": 5000,
            "fisher_subset_seed": 42,
            "fisher_subset": args.fisher_n,
            "fisher_type": "class-marginal AXB K-FAC",
            "factor_source": "AXB-aware SparseGPT arm before each linear module; shared by both arms",
            "B": "all Fisher images, recomputed before each linear module; damping 0.01",
            "A": "current activations before each linear module; damping 0.01",
            "D": "AXB-aware SparseGPT adaptive mask; target fraction imposed separately "
                 "in every 128-column block of every module; identical in both arms",
            "sparsegpt_update": "B compensation in current column, A propagation to later columns",
            "complement_update": "all surviving weights in the same linear module; mu=0; CG rtol=1e-6",
            "canonicalization": "none",
            "biases": "fixed",
            "rho": args.rho,
            "n_sigma": args.n_sigma,
        },
        "targets": targets,
        "parameter_denominator": total,
        "reference_val_loss": base_loss,
        "reference_test_loss": reference_test_loss,
        "runs": [],
    }
    write_json(args.output, result)
    for target in targets:
        sparse_model, complement_model = bm.model(device), bm.model(device)
        modules = []
        for index in range(len(bm.linears(sparse_model))):
            sparse_module = bm.linears(sparse_model)[index]
            complement_module = bm.linears(complement_model)[index]
            with torch.inference_mode():
                activations, gradients = bm.fisher_factors(sparse_model, fisher_images)
                a_input = activations[index][:, :-1]
                covariance = a_input.T @ a_input / len(fisher_images)
                g = gradients[index]
                b = torch.einsum("nci,ncj->ij", g, g) / len(fisher_images)
                b.diagonal().add_(.01 * max(float(b.diag().mean()), torch.finfo(b.dtype).eps))
                b_inverse = torch.linalg.inv(b)
                before = sparse_module.weight.detach().double().clone()
                if not torch.equal(before, complement_module.weight.detach().double()):
                    raise RuntimeError("arms have different starting module weights")
                fisher_sparsegpt_axb(sparse_module, covariance, None,
                                     b_inverse.diag(), target, .01, 128, .01,
                                     b_inverse, b)
                mask = sparse_module.weight.detach() == 0
                a, dead = damped_a(covariance)
                effective_before = before.masked_fill(dead[None, :], 0.)
                sparse_delta = sparse_module.weight.detach().double() - effective_before
                full_delta, diagnostics = complement_update(
                    effective_before, a, b, mask, mu=0., rtol=1e-6,
                    max_iterations=10000)
                complement_module.weight.copy_(
                    (effective_before + full_delta).masked_fill(mask, 0.).to(complement_module.weight))
                actual_full_delta = complement_module.weight.detach().double() - effective_before
                sparse_damage = quadratic(sparse_delta, a, b)
                full_damage = quadratic(actual_full_delta, a, b)
                if full_damage > sparse_damage + 1e-7 * max(sparse_damage, 1e-12):
                    raise RuntimeError(f"full update increased matched quadratic at module {index}")
                modules.append({
                    "module": index,
                    "deleted": int(mask.sum()),
                    "inactive_inputs": int(dead.sum()),
                    "mask_sha256": hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest(),
                    "quadratic": {"sparsegpt": sparse_damage, "full": full_damage},
                    "update_norm": {
                        "sparsegpt": float(sparse_delta.masked_fill(mask, 0.).norm()),
                        "full": float(actual_full_delta.masked_fill(mask, 0.).norm()),
                    },
                    "cg_iterations": diagnostics["iterations"],
                    "cg_relative_residual": diagnostics["relative_residual"],
                })
        arms = {
            "sparsegpt": evaluate(sparse_model, validation, test, reference, total, started, device),
            "full": evaluate(complement_model, validation, test, reference, total, started, device),
        }
        for arm in arms.values():
            arm["admissible"] = arm["val_delta"] <= args.n_sigma * arm["val_se"] + args.rho * base_loss
        if arms["sparsegpt"]["removed"] != arms["full"]["removed"]:
            raise RuntimeError("matched arms have different parameter removal")
        result["runs"].append({"target": target, "arms": arms, "modules": modules})
        write_json(args.output, result)
        print(f"{target:.0%}: removed {arms['full']['frac_removed']:.2%}; "
              f"val sparsegpt={arms['sparsegpt']['val_loss']:.6f}, "
              f"full={arms['full']['val_loss']:.6f}; "
              f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
    result["status"] = "complete"
    first_rejected = next((index for index, run in enumerate(result["runs"])
                           if not run["arms"]["full"]["admissible"]), None)
    result["stop_index"] = (first_rejected if first_rejected is not None
                            else len(result["runs"]))
    result["termination_reason"] = ("stop_rule" if first_rejected is not None
                                    else "targets_exhausted")
    result["seconds"] = time.perf_counter() - started
    write_json(args.output, result)
