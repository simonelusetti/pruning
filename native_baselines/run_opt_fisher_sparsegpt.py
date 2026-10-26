#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))
from nnprune.canonicalize.mlp import canonicalize_linear_chain
from nnprune.factorized_fisher import complement_update, inverse_diagonal
from nnprune.fisher import PRESETS, resolve_strategy, validate_strategy

from result_artifacts import write_json
from run_cpu_close import (calibration_tokens, linear_layers, per_sequence_loss,
                           selection_sequences)


class StopCapture(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--model-dtype", choices=("auto", "float32", "float64", "float16", "bfloat16"),
                        default="auto")
    parser.add_argument("--targets", default=".02,.05,.1")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--b-sequences", type=int, default=8)
    parser.add_argument("--b-context", type=int, default=128)
    parser.add_argument("--b-tokens", type=int, default=256)
    parser.add_argument("--b-batch-size", type=int, default=1)
    parser.add_argument("--b-damp-ratio", type=float, default=1e-6)
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--modules", default="all")
    parser.add_argument("--selection-sequences", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=12345)
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--fisher-preset", choices=PRESETS,
                        default="fisher_sparsegpt")
    parser.add_argument("--fisher-geometry", choices=("axb", "axi"))
    parser.add_argument("--fisher-deletion", choices=("obs", "miqp"))
    parser.add_argument("--fisher-compensation",
                        choices=("sparsegpt", "sparse_miqp", "dense", "complement"))
    parser.add_argument("--fisher-cg-rtol", type=float, default=1e-6)
    parser.add_argument("--fisher-cg-max-iterations", type=int, default=1000)
    parser.add_argument("--fisher-update-scope", choices=("layer",))
    parser.add_argument("--fisher-structure", choices=("post", "none"))
    parser.add_argument("--fisher-decision", choices=("validate", "proxy"))
    parser.add_argument("--fisher-lambda-u", type=float, default=1e-8)
    parser.add_argument("--fisher-lambda-d", type=float, default=1e-7)
    parser.add_argument("--fisher-mu", type=float, default=1e-4)
    parser.add_argument("--fisher-pool", type=int, default=128)
    parser.add_argument("--fisher-repeat", action="store_true",
                        help="repeat bounded layer MIQPs to reach each target")
    parser.add_argument("--fisher-time-limit", type=float, default=10.)
    parser.add_argument("--fisher-gap", type=float, default=.05)
    parser.add_argument("--fisher-bound", type=float, default=10.)
    parser.add_argument("--fisher-log", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "native_baselines/results/opt_published/fisher_sparsegpt.json")
    args = parser.parse_args()
    args.targets = tuple(float(value) for value in args.targets.split(","))
    positive = (args.nsamples, args.seqlen, args.b_sequences, args.b_context,
                args.b_tokens, args.b_batch_size, args.blocksize, args.max_layers,
                args.selection_sequences, args.threads)
    if min(positive) < 1 or args.b_sequences > args.nsamples:
        parser.error("sample and block settings must be positive")
    if (not args.targets or tuple(sorted(args.targets)) != args.targets or
            any(not 0 < value <= 1 for value in args.targets) or
            (1. in args.targets and not args.fisher_repeat)):
        parser.error("targets must be increasing fractions in (0, 1]; 1 requires --fisher-repeat")
    if args.resume and args.fisher_repeat:
        parser.error("repeated MIQP runs cannot resume without a saved model checkpoint")
    if min(args.b_damp_ratio, args.percdamp, args.rho, args.n_sigma) < 0:
        parser.error("damping must be non-negative")
    if min(args.fisher_lambda_u, args.fisher_lambda_d, args.fisher_mu) < 0:
        parser.error("Fisher penalties must be non-negative")
    if not 0 < args.fisher_cg_rtol < 1 or args.fisher_cg_max_iterations < 1:
        parser.error("invalid complement CG convergence settings")
    if (args.fisher_pool < 1 or args.fisher_time_limit <= 0 or
            not 0 <= args.fisher_gap < 1 or args.fisher_bound <= 0):
        parser.error("invalid Fisher MIQP limit")
    try:
        args.fisher_strategy = validate_strategy(resolve_strategy(
            args.fisher_preset, geometry=args.fisher_geometry,
            deletion=args.fisher_deletion, compensation=args.fisher_compensation,
            update_scope=args.fisher_update_scope,
            structure=args.fisher_structure, decision=args.fisher_decision), "opt")
    except ValueError as error:
        parser.error(str(error))
    return args


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
        predicted += .5 * float(errors.square().sum() if b is None else
                                (errors * (b @ errors)).sum())
        weight[:, start:end] = result
        weight[:, end:] -= errors @ inverse_factor[start:end, end:]
    module.weight.copy_(weight.to(module.weight))
    return {"deleted": int((weight == 0).sum()),
            "predicted_fisher_damage": predicted}


@torch.inference_mode()
def fisher_sparsegpt_axb(module, a, gradients, b_inverse_diagonal, sparsity,
                         percdamp, blocksize, b_damp_ratio,
                         b_inverse_matrix=None, b_factor=None):
    device = module.weight.device
    weight = module.weight.detach().to(device=device, dtype=torch.float32).clone()
    a = ((a + a.T) / 2).to(device=device, dtype=torch.float32)
    dead = a.diag() == 0
    a[dead, dead] = 1
    weight[:, dead] = 0
    a.diagonal().add_(percdamp * a.diag().mean())
    factor, info = torch.linalg.cholesky_ex(a)
    if int(info):
        raise RuntimeError("damped activation factor is not positive definite")
    inverse_factor = torch.linalg.cholesky(
        torch.cholesky_inverse(factor), upper=True)

    if b_inverse_matrix is None:
        gradients = gradients.to(device=device, dtype=torch.float64)
        scaled = gradients / math.sqrt(len(gradients))
        damping = b_damp_ratio * max(
            float(scaled.square().sum(0).mean()), torch.finfo(torch.float64).eps)
        if damping <= 0:
            raise ValueError("B-aware column compensation requires positive damping")
    else:
        b_inverse_matrix = b_inverse_matrix.to(device=device, dtype=torch.float64)
        b_factor = b_factor.to(device=device, dtype=torch.float64)
    b_inverse_diagonal = b_inverse_diagonal.to(device=device)
    predicted = 0.
    for start in range(0, weight.shape[1], blocksize):
        end = min(start + blocksize, weight.shape[1])
        current = weight[:, start:end].clone()
        result = torch.zeros_like(current)
        errors = torch.zeros_like(current)
        local_inverse = inverse_factor[start:end, start:end]
        denominator = (b_inverse_diagonal[:, None] *
                       local_inverse.diag().double().square()[None, :]).clamp_min(1e-30)
        score = current.double().square() / denominator
        count = min(int(score.numel() * sparsity), score.numel() - 1)
        threshold = torch.kthvalue(score.flatten(), count + 1).values
        mask = score <= threshold
        for column in range(end - start):
            deleted = mask[:, column]
            values = current[:, column]
            change = torch.zeros_like(values)
            change[deleted] = -values[deleted]
            if b_inverse_matrix is not None and bool(deleted.any()):
                indices = torch.where(deleted)[0]
                subset = b_inverse_matrix[indices[:, None], indices]
                multipliers = torch.linalg.solve(subset, change[deleted].double())
                change = (b_inverse_matrix[:, indices] @ multipliers).to(change)
            elif b_inverse_matrix is None and bool(deleted.any()) and bool((~deleted).any()):
                live_gradients = scaled[:, ~deleted]
                deleted_effect = scaled[:, deleted] @ change[deleted].double()
                gram = live_gradients @ live_gradients.T
                gram.diagonal().add_(damping)
                correction = torch.linalg.solve(gram, deleted_effect)
                change[~deleted] = -(live_gradients.T @ correction).to(change)
            quantized = values + change
            quantized[deleted] = 0
            result[:, column] = quantized
            error = (values - quantized) / local_inverse[column, column]
            current[:, column:] -= error[:, None] @ local_inverse[
                column, column:][None, :]
            errors[:, column] = error
        predicted += .5 * float(
            (scaled @ errors.double()).square().sum() +
            damping * errors.double().square().sum()
            if b_inverse_matrix is None else
            (errors.double() * (b_factor @ errors.double())).sum())
        weight[:, start:end] = result
        weight[:, end:] -= errors @ inverse_factor[start:end, end:]
    module.weight.copy_(weight.to(module.weight))
    return {"deleted": int((weight == 0).sum()),
            "predicted_fisher_damage": predicted}


@torch.inference_mode()
def fisher_layer_complement(module, a, b, b_inverse_diagonal, sparsity, args):
    started = time.perf_counter()
    weight = module.weight.detach().double().clone()
    a = a.to(weight)
    a = ((a + a.T) / 2).clone()
    a.diagonal().add_(args.percdamp * a.diag().mean())
    b = b.to(weight)
    deleted = weight == 0
    count = max(0, math.ceil(sparsity * weight.numel()) - int(deleted.sum()))
    if count:
        saliency = weight.square() / (
            b_inverse_diagonal.to(weight)[:, None] *
            inverse_diagonal(a).to(weight)[None, :]).clamp_min(1e-30)
        saliency.masked_fill_(deleted, torch.inf)
        indices = torch.topk(saliency.flatten(), count, largest=False).indices
        deleted.flatten()[indices] = True
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
            "deletion_selection": "fixed generalized-OBS ranking; no joint support optimization"}


@torch.inference_mode()
def fisher_layer_miqp(module, a, b, b_inverse_diagonal, sparsity, args,
                      *, prepared=False, deletion_count=None):
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError("layer-local Fisher MIQP requires gurobipy") from error

    weight = module.weight.detach().double().cpu().clone()
    a = ((a.double() + a.double().T) / 2).clone()
    if not prepared:
        a.diagonal().add_(args.percdamp * a.diag().mean())
    b = ((b.double() + b.double().T) / 2).clone()
    a_inverse_diagonal = (args._a_inverse_diagonal if prepared else inverse_diagonal(a))
    saliency = weight.square() / (
        b_inverse_diagonal[:, None] * a_inverse_diagonal[None, :]).clamp_min(1e-30)
    live = weight != 0
    live_indices = live.flatten().nonzero().flatten()
    pool_size = min(args.fisher_pool, len(live_indices))
    deletion_ids = live_indices[torch.topk(
        saliency.flatten()[live_indices], pool_size, largest=False).indices]
    maximum_deletions = (max(1, math.ceil(sparsity * weight.numel()))
                         if deletion_count is None else deletion_count)
    if maximum_deletions > len(deletion_ids):
        raise ValueError(
            f"target {sparsity:g} requires {maximum_deletions} deletion candidates "
            f"for a {tuple(weight.shape)} layer, but fisher_pool={args.fisher_pool}")

    width = weight.shape[1]
    deletion_rows = torch.div(deletion_ids, width, rounding_mode="floor")
    deletion_columns = deletion_ids % width
    deleted_weights = weight.flatten()[deletion_ids]
    correlation = ((b[:, deletion_rows] * deleted_weights[None]) @
                   a[:, deletion_columns].T)
    update_scale = (b.diag()[:, None] * a.diag()[None] +
                    args.fisher_mu).sqrt().clamp_min(1e-30)
    update_score = correlation.abs() / update_scale
    update_score.flatten()[~live.flatten()] = -1
    update_score.flatten()[deletion_ids] = -1
    update_ids = torch.topk(update_score.flatten(),
                            min(args.fisher_pool, len(live_indices) - len(deletion_ids))).indices

    coordinates = torch.cat((deletion_ids, update_ids))
    rows = torch.div(coordinates, width, rounding_mode="floor")
    columns = coordinates % width
    q = b[rows[:, None], rows[None]] * a[columns[:, None], columns[None]]
    q = (q + q.T) / 2
    if args.fisher_strategy.compensation == "dense":
        return _fisher_layer_miqp_dense(
            module, weight, q, deletion_ids, update_ids, deleted_weights,
            maximum_deletions, deletion_count, args, gp)
    values, vectors = torch.linalg.eigh(q)
    q = (vectors * values.clamp_min(0)) @ vectors.T
    relative_lambda_u = getattr(args, "fisher_lambda_u_relative", None)
    lambda_u = (args.fisher_lambda_u if relative_lambda_u is None else
                relative_lambda_u * .5 *
                float(saliency.flatten()[deletion_ids].median()))

    solver = gp.Model("fisher_layer")
    solver.Params.OutputFlag = int(args.fisher_log)
    solver.Params.TimeLimit = args.fisher_time_limit
    solver.Params.MIPGap = args.fisher_gap
    sparse = args.fisher_strategy.compensation == "sparse_miqp"
    d = [solver.addVar(vtype=gp.GRB.BINARY, name=f"d_{int(index)}")
         for index in deletion_ids]
    median = float(weight[live].abs().median())
    u, c, bounds = [], [], []
    for position, index in enumerate(update_ids):
        bound = args.fisher_bound * max(abs(float(weight.flatten()[index])), median, 1e-6)
        update = (solver.addVar(vtype=gp.GRB.BINARY, name=f"u_{int(index)}")
                  if sparse else None)
        compensation = solver.addVar(lb=-bound, ub=bound, name=f"c_{int(index)}")
        if sparse:
            solver.addGenConstrIndicator(update, 0, compensation == 0.)
        u.append(update)
        c.append(compensation)
        bounds.append(bound)
    if deletion_count is None:
        solver.addConstr(gp.quicksum(d) >= 1)
        solver.addConstr(gp.quicksum(d) <= maximum_deletions)
    else:
        solver.addConstr(gp.quicksum(d) == maximum_deletions)

    scale = max(float(q.diag().max()), args.fisher_mu,
                lambda_u, args.fisher_lambda_d, 1e-30)
    multipliers = torch.cat((-deleted_weights, torch.ones(len(c), dtype=q.dtype)))
    quadratic = q * multipliers[:, None] * multipliers[None, :] / scale
    quadratic.diagonal()[len(d):].add_(args.fisher_mu / scale)
    decisions = gp.MVar.fromlist(d + c)
    objective = .5 * decisions @ quadratic.numpy() @ decisions
    if sparse:
        objective += (lambda_u / scale) * gp.quicksum(u)
    objective -= (args.fisher_lambda_d / scale) * gp.quicksum(d)
    solver.setObjective(objective, gp.GRB.MINIMIZE)
    for position, variable in enumerate(d):
        variable.Start = int(position < (maximum_deletions if deletion_count is not None else 1))
    for update, compensation in zip(u, c):
        if sparse:
            update.Start = 0
        compensation.Start = 0.
    solver.optimize()
    if solver.SolCount == 0:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no layer-local solution (status {status})")

    deleted_positions = [position for position, value in enumerate(d) if value.X > .5]
    updated_positions = [position for position, value in enumerate(c)
                         if (u[position].X > .5 if sparse else value.X != 0.)]
    delta = torch.zeros(len(coordinates), dtype=torch.float64)
    delta[deleted_positions] = -deleted_weights[deleted_positions]
    if updated_positions:
        delta[len(deletion_ids) + torch.tensor(updated_positions)] = torch.tensor(
            [c[position].X for position in updated_positions], dtype=delta.dtype)
    damage = float(.5 * delta @ q @ delta)
    update_norm = float(delta[len(deletion_ids):].square().sum())
    direct = (damage + .5 * args.fisher_mu * update_norm +
              (lambda_u * len(updated_positions) if sparse else 0.) -
              args.fisher_lambda_d * len(deleted_positions))
    solver_objective = float(solver.ObjVal) * scale
    tolerance = 1e-7 * max(1., abs(direct), abs(solver_objective))
    if abs(direct - solver_objective) > tolerance:
        solver.dispose()
        raise RuntimeError("layer-local MIQP objective disagrees with direct evaluation")

    flat = weight.flatten()
    for position in updated_positions:
        flat[update_ids[position]] += c[position].X
    flat[deletion_ids[deleted_positions]] = 0
    gap, seconds, status = float(solver.MIPGap), float(solver.Runtime), int(solver.Status)
    bound_active = any(abs(c[position].X) >= .999 * bounds[position]
                       for position in updated_positions)
    solver.dispose()
    module.weight.copy_(weight.to(module.weight))
    return {
        "deleted": len(deleted_positions),
        "updates": len(updated_positions),
        "predicted_fisher_damage": damage,
        "search_objective": direct,
        "solver_objective": solver_objective,
        "solver_gap": gap,
        "solver_seconds": seconds,
        "solver_status": status,
        "compensation_bound_active": bound_active,
        "candidate_deletions": len(deletion_ids),
        "candidate_updates": len(update_ids),
        "candidate_coverage": len(deletion_ids) / weight.numel(),
        "maximum_deletions": maximum_deletions,
        "lambda_u": lambda_u,
    }


def _fisher_layer_miqp_dense(module, weight, q, deletion_ids, update_ids,
                             deleted_weights, maximum_deletions,
                             deletion_count, args, gp):
    count = len(deletion_ids)
    q_dd = q[:count, :count]
    q_du = q[:count, count:]
    q_uu = q[count:, count:].clone()
    q_uu.diagonal().add_(args.fisher_mu)
    response = torch.linalg.solve(q_uu, q_du.T)
    reduced = q_dd - q_du @ response
    reduced = (reduced + reduced.T) / 2
    multiplier = -deleted_weights
    binary_quadratic = reduced * multiplier[:, None] * multiplier[None, :]
    binary_quadratic = (binary_quadratic + binary_quadratic.T) / 2
    scale = max(float(binary_quadratic.abs().max()),
                args.fisher_lambda_d, 1e-30)

    solver = gp.Model("fisher_layer_dense")
    solver.Params.OutputFlag = int(args.fisher_log)
    solver.Params.TimeLimit = args.fisher_time_limit
    solver.Params.MIPGap = args.fisher_gap
    decisions = solver.addMVar(count, vtype=gp.GRB.BINARY)
    if deletion_count is None:
        solver.addConstr(decisions.sum() >= 1)
        solver.addConstr(decisions.sum() <= maximum_deletions)
    else:
        solver.addConstr(decisions.sum() == maximum_deletions)
    objective = (.5 * decisions @ (binary_quadratic / scale).numpy() @ decisions -
                 (args.fisher_lambda_d / scale) * decisions.sum())
    solver.setObjective(objective, gp.GRB.MINIMIZE)
    decisions.Start = [int(position < maximum_deletions)
                       for position in range(count)]
    solver.optimize()
    if solver.SolCount == 0:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no dense layer-local solution (status {status})")

    selected = torch.as_tensor(decisions.X > .5)
    deletion_delta = multiplier * selected
    compensation = -(response @ deletion_delta)
    delta = torch.cat((deletion_delta, compensation))
    damage = float(.5 * delta @ q @ delta)
    update_norm = float(compensation.square().sum())
    direct = (damage + .5 * args.fisher_mu * update_norm -
              args.fisher_lambda_d * int(selected.sum()))
    solver_objective = float(solver.ObjVal) * scale
    tolerance = 1e-6 * max(abs(direct), abs(solver_objective), 1e-15)
    if abs(direct - solver_objective) > tolerance:
        solver.dispose()
        raise RuntimeError("dense MIQP Schur objective disagrees with direct evaluation")

    live = weight != 0
    median = float(weight[live].abs().median())
    bounds = args.fisher_bound * torch.maximum(
        weight.flatten()[update_ids].abs(),
        torch.full_like(compensation, max(median, 1e-6)))
    flat = weight.flatten()
    flat[update_ids] += compensation
    flat[deletion_ids[selected]] = 0
    gap, seconds, status = float(solver.MIPGap), float(solver.Runtime), int(solver.Status)
    solver.dispose()
    module.weight.copy_(weight.to(module.weight))
    return {
        "deleted": int(selected.sum()),
        "updates": int((compensation.abs() > 1e-12).sum()),
        "predicted_fisher_damage": damage,
        "search_objective": direct,
        "solver_objective": solver_objective,
        "solver_gap": gap,
        "solver_seconds": seconds,
        "solver_status": status,
        "compensation_bound_active": False,
        "reference_bound_exceeded": bool((compensation.abs() >= bounds).any()),
        "candidate_deletions": count,
        "candidate_updates": len(update_ids),
        "candidate_coverage": count / weight.numel(),
        "maximum_deletions": maximum_deletions,
        "lambda_u": 0.,
        "update_norm": update_norm,
    }


@torch.inference_mode()
def fisher_layer_miqp_repeated(module, a, b, b_inverse_diagonal, sparsity, args):
    from copy import copy

    local_args = copy(args)
    a = ((a.double() + a.double().T) / 2).clone()
    a.diagonal().add_(args.percdamp * a.diag().mean())
    local_args._a_inverse_diagonal = inverse_diagonal(a)
    target = math.ceil(sparsity * module.weight.numel())
    remaining = max(0, target - int((module.weight == 0).sum()))
    totals = {"deleted": 0, "updates": 0, "solver_seconds": 0.,
              "predicted_fisher_damage": 0., "search_objective": 0.,
              "solver_objective": 0.}
    moves, max_gap, bound_active, reference_bound_exceeded = 0, 0., False, False
    while remaining:
        # Leave alternatives in the pool so the MIQP still chooses deletions.
        count = min(remaining, max(1, args.fisher_pool // 2))
        record = fisher_layer_miqp(
            module, a, b, b_inverse_diagonal, sparsity, local_args,
            prepared=True, deletion_count=count)
        for key in totals:
            totals[key] += record[key]
        moves += 1
        remaining = max(0, target - int((module.weight == 0).sum()))
        max_gap = max(max_gap, record["solver_gap"])
        bound_active |= record["compensation_bound_active"]
        reference_bound_exceeded |= record.get("reference_bound_exceeded", False)
        if moves == 1 or moves % 1000 == 0 or remaining == 0:
            print(f"MIQP {tuple(module.weight.shape)}: move {moves}, "
                  f"{remaining} deletions remaining, max gap {max_gap:g}", flush=True)
    return {**totals, "moves": moves, "solver_gap": max_gap,
            "compensation_bound_active": bound_active,
            "reference_bound_exceeded": reference_bound_exceeded,
            "target_zeros": target, "actual_zeros": int((module.weight == 0).sum()),
            "damage_aggregation": "sum of per-move quadratics; excludes cross-move terms"}


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
                names, args, progress):
    started = time.perf_counter()
    hidden, kwargs = capture_first_inputs(model, calibration)
    history, canonical_savings = [], 0
    for layer_index, layer in enumerate(model.model.decoder.layers[:layers_count]):
        layer_started = time.perf_counter()
        a_factors = collect_a(layer, hidden, kwargs, names)
        modules = linear_layers(layer)
        records = {}
        for name in names:
            key = (layer_index, name)
            if args.fisher_strategy.compensation == "complement":
                records[name] = fisher_layer_complement(
                    modules[name], a_factors[name], b_factors[key], b_inverse[key],
                    target, args)
            elif args.fisher_strategy.deletion == "miqp":
                update = (fisher_layer_miqp_repeated if args.fisher_repeat
                          else fisher_layer_miqp)
                records[name] = update(
                    modules[name], a_factors[name], b_factors[key], b_inverse[key],
                    target, args)
            else:
                records[name] = fisher_sparsegpt(
                    modules[name], a_factors[name],
                    None if b_factors is None else b_factors[key],
                    None if b_inverse is None else b_inverse[key],
                    target, args.percdamp, args.blocksize)
        before = nonzero_count(model, layer_index + 1, names)
        if (args.fisher_strategy.structure == "post" and
                "fc1" in names and "fc2" in names):
            canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))
        after = nonzero_count(model, layer_index + 1, names)
        canonical_savings += max(0, before - after)
        hidden = forward_layer(layer, hidden, kwargs)
        history.append({"layer": layer_index, "modules": records,
                        "canonical_savings": canonical_savings,
                        "seconds": time.perf_counter() - layer_started})
        progress(layer_index + 1, history, canonical_savings,
                 time.perf_counter() - started)
        print(f"target {target:.0%}, layer {layer_index + 1}/{layers_count}: "
              f"{history[-1]['seconds']:.1f}s", flush=True)
    return history, canonical_savings, time.perf_counter() - started


def main():
    args = parse_args()
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
    print("Loading reference model", flush=True)
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=args.model_dtype).to(device).eval()
    reference_model.config.use_cache = False
    layers_count = min(args.max_layers, len(reference_model.model.decoder.layers))
    names = selected_names(reference_model.model.decoder.layers[0], args.modules)
    reference_loss = per_sequence_loss(reference_model, diagnostic).double()
    print(f"Reference loss: {float(reference_loss.mean()):.6f}", flush=True)
    baseline = float(reference_loss.mean())
    denominator = parameter_count(reference_model, layers_count, names)
    initial_nonzero = nonzero_count(reference_model, layers_count, names)
    if args.fisher_strategy.geometry == "axb":
        b_sequences = [value[:, :min(args.seqlen, args.b_context)]
                       for value in calibration[:args.b_sequences]]
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
    del reference_model

    strategy = args.fisher_strategy.as_dict()
    b_description = ("dense, estimated once from the original model"
                     if args.fisher_strategy.geometry == "axb" else "identity")
    approximations = ["layerwise K-FAC uses F_W approximately equal to A kron B"]
    if args.fisher_strategy.geometry == "axb":
        approximations += [
            "B is frozen at the original model",
            "sampled token score gradients omit most cross-token covariance",
        ]
    complement = args.fisher_strategy.compensation == "complement"
    if complement:
        approximations += [
            "deletions are fixed by generalized-OBS ranking, not joint MIQP selection",
            "all surviving module weights compensate via a matrix-free constrained quadratic solve",
            "A/B stay fixed within each module; biases remain fixed",
        ]
    elif args.fisher_strategy.deletion == "miqp":
        approximations += [
            "each layer-local MIQP is restricted to Fisher-ranked deletion and update pools",
            ("bounded MIQP moves refresh candidates and weights until the target; A/B remain fixed within each module"
             if args.fisher_repeat else
             "one MIQP move is applied per eligible linear module without refreshing A within the transformer layer"),
            "OPT bias parameters remain fixed",
        ]
        if args.fisher_strategy.structure == "post":
            approximations.append(
                "canonical savings are applied after each transformer layer and are not rewarded in the MIQP")
    else:
        approximations += [
            "the block mask uses scalar generalized-OBS scores and ignores interactions among simultaneous deletions",
            "SparseGPT fixes each processed column, so compensation is unrestricted only in later input columns",
        ]

    result = {
        "result_schema": 3, "status": "running",
        "method": ("Fisher full-layer complement" if complement else
                   "Fisher layer-local MIQP" if args.fisher_strategy.deletion == "miqp"
                   else "Fisher-aware SparseGPT"),
        "model": args.model, "model_dtype": args.model_dtype,
        "fisher_strategy": strategy,
        "calibration": {"dataset": "c4", "nsamples": args.nsamples,
                        "seqlen": args.seqlen, "seed": args.seed,
                        "b_sequences": args.b_sequences,
                        "b_context": args.b_context, "b_tokens": args.b_tokens},
        "protocol": {"blocksize": args.blocksize, "percdamp": args.percdamp,
                     "b_damp_ratio": args.b_damp_ratio,
                     "B": b_description,
                     "A": "collected once per layer from the current layer inputs",
                     "mask": ("generalized OBS candidate pool"
                              if args.fisher_strategy.deletion == "miqp"
                              else "generalized OBS saliency"),
                     "compensation": ("full-complement constrained AXB solve" if complement else
                                      "Gurobi MIQP within one linear layer"
                                      if args.fisher_strategy.deletion == "miqp"
                                      else "SparseGPT columnwise error propagation"),
                     "fisher_pool": args.fisher_pool,
                     "fisher_repeat": args.fisher_repeat,
                     "fisher_time_limit": args.fisher_time_limit,
                     "fisher_gap": args.fisher_gap,
                     "fisher_lambda_u": args.fisher_lambda_u,
                     "fisher_lambda_d": args.fisher_lambda_d,
                     "fisher_mu": args.fisher_mu,
                     "fisher_cg_rtol": args.fisher_cg_rtol,
                     "fisher_cg_max_iterations": args.fisher_cg_max_iterations,
                     "fisher_bound": args.fisher_bound},
        "approximations": approximations,
        "b_estimation": {**b_timing, "off_diagonal_frobenius_ratio": off_diagonal},
        "parameter_denominator": denominator,
        "selection": {"dataset": "c4", "seed": args.selection_seed,
                      "count": args.selection_sequences, "rho": args.rho,
                      "n_sigma": args.n_sigma},
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
    write_json(args.output, result, "opt")
    model = None
    for target in args.targets:
        if target in completed_targets:
            continue
        print(f"Starting target {target:.0%}", flush=True)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=args.model_dtype).to(device).eval()
        model.config.use_cache = False
        run = {"target_sparsity": target, "status": "running", "layers": []}
        result["runs"].append(run)

        def progress(completed, history, canonical_savings, seconds):
            run.update({"completed_layers": completed, "layers": history,
                        "canonical_savings": canonical_savings,
                        "seconds": seconds})
            write_json(args.output, result)

        try:
            history, canonical_savings, seconds = prune_model(
                model, calibration, target, b_factors, b_inverse, layers_count,
                names, args, progress)
        except Exception as error:
            run.update(status="failed", error=str(error))
            result.update(status="failed", termination_reason="solver_or_pruning_error",
                          error=str(error))
            write_json(args.output, result, "opt")
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
        }
        row["admissible"] = (row["selection_delta"] <=
                             args.n_sigma * row["selection_se"] +
                             args.rho * baseline)
        result["curve"].append(row)
        run.update({"status": "complete", "layers": history, **row,
                    "peak_memory_mb": peak_memory_mb(device)})
        write_json(args.output, result, "opt")
        if not args.fisher_repeat:
            del model
            model = None
        if args.fisher_strategy.decision == "validate" and not row["admissible"]:
            result["termination_reason"] = "loss_limit"
            break
    else:
        result["termination_reason"] = "targets_exhausted"
    result["status"] = "complete"
    write_json(args.output, result, "opt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
