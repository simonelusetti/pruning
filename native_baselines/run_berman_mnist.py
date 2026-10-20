#!/usr/bin/env python3
"""Berman SimpleNN/MNIST: Berman iterative vs classic + weight zeroing vs compensated Fisher.

Every Fisher deletion solves the same problem.  D = parameters set to zero, R = live
parameters that compensate, F = model Fisher (sum over classes, not empirical):
    delta_R = F_RR^+ F_RD theta_D,    cost = 1/2 delta^T F delta,  delta_D = -theta_D.
Weights: D = one scalar, R = a block of its output row (including the row bias), or empty.
Neurons: D = the neuron's outgoing weights, R = all other outgoing weights + next bias.
Our methods choose on valA, stop on valB, and count parameters after canonicalisation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))
from nnprune.canonicalize.mlp import (canonicalize_linear_chain,
                                      canonicalize_mlp_tensors)
from nnprune.third_party.fdist_fim import calculate_fim_backprop

DATA = ROOT / "native_baselines" / "cache" / "berman" / "mnist"
CHECKPOINT = ROOT / "checkpoints" / "berman" / "simplenn_mnist_official.pt"
RESULTS = ROOT / "native_baselines" / "results_berman" / "simplenn_mnist"
SPLIT_SEED = 12345
COMMON = {
    "result_schema": 3,
    "paper": "Berman et al. (2026), arXiv:2609.16129",
    "source_commit": "8d60ea99027a284f8f02834e8d7d5c72bc278cda",
    "model": "SimpleNN_h2_n64", "dataset": "MNIST",
    "checkpoint": "official released checkpoint",
}


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x.flatten(1))


def _idx(path, images):
    with gzip.open(path, "rb") as stream:
        if images:
            _, n, h, w = struct.unpack(">IIII", stream.read(16))
            value = torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8)
            return value.reshape(n, 1, h, w).float().div_(255).sub_(.1307).div_(.3081)
        _, n = struct.unpack(">II", stream.read(8))
        return torch.frombuffer(bytearray(stream.read()), dtype=torch.uint8).long()


def mnist():
    train = (_idx(DATA / "train-images-idx3-ubyte.gz", True),
             _idx(DATA / "train-labels-idx1-ubyte.gz", False))
    test = (_idx(DATA / "t10k-images-idx3-ubyte.gz", True),
            _idx(DATA / "t10k-labels-idx1-ubyte.gz", False))
    return train, test


def loader(data, batch_size):
    return DataLoader(TensorDataset(*data), batch_size=batch_size)


def model(device):
    net = SimpleNN().to(device)
    net.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    return net.eval().requires_grad_(False)


def linears(net):
    return net.net[0], net.net[2], net.net[4]


@torch.inference_mode()
def losses(net, batches, device):
    return torch.cat([F.cross_entropy(net(x.to(device)), y.to(device), reduction="none").cpu()
                      for x, y in batches])


@torch.inference_mode()
def accuracy(net, batches, device):
    correct = sum(int((net(x.to(device)).argmax(1) == y.to(device)).sum()) for x, y in batches)
    return correct / sum(len(y) for _, y in batches)


def paired(candidate, reference):
    delta = (candidate - reference).double()
    return float(delta.mean()), float(delta.std() / math.sqrt(delta.numel()))


def nonzero(net):
    return sum(int(torch.count_nonzero(p)) for p in net.parameters())


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    os.replace(temporary, path)


def evaluate_row(net, val, test, reference, total, started, step, device, **extra):
    val_loss = losses(net, val, device)
    delta, se = paired(val_loss, reference)
    removed = total - nonzero(net)
    return {"step": step, "removed": removed, "frac_removed": removed / total,
            "val_loss": float(val_loss.mean()), "val_delta": delta, "val_se": se,
            "test_loss": float(losses(net, test, device).mean()),
            "test_acc": accuracy(net, test, device),
            "seconds": time.perf_counter() - started, **extra}


def protocol(args, **extra):
    return {"split_seed": SPLIT_SEED, "val_a": 2500, "val_b": 2500, "test": 5000,
            "rho": args.rho, "n_sigma": args.n_sigma, "fisher_subset_seed": 42, **extra}


def fisher_images(train, n):
    order = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(42))[:n]
    return train[0][order], train[1][order]


# --------------------------------------------------------------------------
# Berman et al.: iterative sqrt(F_ii)|theta_i| on a 10% grid, their parameter count
# --------------------------------------------------------------------------

@torch.inference_mode()
def prune_lowest(net, scores, count):
    params = list(net.parameters())
    flat = torch.cat([torch.where(p != 0, s, torch.inf).flatten()
                      for p, s in zip(params, scores)])
    chosen = torch.topk(flat, count, largest=False).indices
    mask = torch.ones_like(flat)
    mask[chosen] = 0
    offset = 0
    for p in params:
        p.mul_(mask[offset:offset + p.numel()].view_as(p))
        offset += p.numel()


def run_berman(args, train, val_b, test, n_fisher):
    net = model(args.device)
    total = sum(p.numel() for p in net.parameters())
    reference = losses(net, val_b, args.device)
    fisher_data = loader(fisher_images(train, n_fisher), 128)
    started = time.perf_counter()
    rows = [evaluate_row(net, val_b, test, reference, total, started, 0, args.device)]
    for step in range(1, min(10, args.max_steps) + 1):
        diag = calculate_fim_backprop(net, fisher_data, device=str(args.device)).to(args.device)
        scores, offset = [], 0
        for p in net.parameters():
            scores.append(diag[offset:offset + p.numel()].view_as(p).clamp_min(0).sqrt()
                          * p.detach().abs())
            offset += p.numel()
        prune_lowest(net, scores, round(step / 10 * total) - (total - nonzero(net)))
        rows.append(evaluate_row(net, val_b, test, reference, total, started, step,
                                 args.device))
        print(f"Berman n={n_fisher} {step * 10}%: val {rows[-1]['val_loss']:.4f}", flush=True)
    base = rows[0]["val_loss"]
    failed = [i for i, row in enumerate(rows)
              if row["val_delta"] > args.n_sigma * row["val_se"] + args.rho * base]
    stop_index = (failed[0] - 1) if failed else len(rows) - 1
    name = "berman_iterative.json" if n_fisher == args.fisher_n else \
        f"berman_iterative_n{n_fisher}.json"
    save(RESULTS / name, {
        **COMMON, "status": "complete", "method": "fdist_iterative",
        "protocol": protocol(args, fisher_subset=n_fisher, grid="10%"),
        "stop_index": stop_index, "stopping_point": rows[stop_index], "history": rows})


def run_berman_validated(args, train, val_a, val_b, test):
    net = model(args.device)
    total = sum(p.numel() for p in net.parameters())
    ref_b = losses(net, val_b, args.device)
    base = float(ref_b.mean())
    fisher_data = loader(fisher_images(train, args.fisher_n), 128)
    started = time.perf_counter()
    rows = [evaluate_row(net, val_b, test, ref_b, total, started, 0, args.device)]
    stop_index, rejected, termination_reason = None, None, "schedule_complete"
    for step in range(1, min(10, args.max_steps) + 1):
        diag = calculate_fim_backprop(net, fisher_data, device=str(args.device)).to(args.device)
        params = list(net.parameters())
        flat_theta = torch.cat([parameter.detach().flatten() for parameter in params])
        score = diag.clamp_min(0).sqrt() * flat_theta.abs()
        score = score.masked_fill(flat_theta == 0, torch.inf)
        count = round(step / 10 * total) - (total - nonzero(net))
        ranked = torch.argsort(score)
        chunks = [chunk for chunk in ranked.split(count)[:args.berman_candidate_pool]
                  if len(chunk) == count and torch.isfinite(score[chunk]).all()]
        if not chunks:
            termination_reason = "candidate_exhaustion"
            break
        state, evaluated = snapshot(net), []
        for chunk in chunks:
            mask = torch.ones_like(flat_theta)
            mask[chunk] = 0
            offset = 0
            for parameter in params:
                parameter.mul_(mask[offset:offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
            evaluated.append((chunk, losses(net, val_a, args.device)))
            net.load_state_dict(state)
        best = min(evaluated, key=lambda item: float(item[1].mean()))[1]
        ties = [item for item in evaluated
                if paired(item[1], best)[0] <= args.n_sigma * paired(item[1], best)[1]]
        chosen, chosen_loss = min(ties, key=lambda item: float(item[1].mean()))
        flat = torch.cat([parameter.flatten() for parameter in params])
        flat[chosen] = 0
        offset = 0
        for parameter in params:
            parameter.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()
        delta, se = paired(losses(net, val_b, args.device), ref_b)
        extra = {"kind": "berman_validated", "move": f"berman_chunk:{len(chosen)}",
                 "actual_savings": len(chosen), "val_a_loss": float(chosen_loss.mean()),
                 "n_candidates": len(evaluated), "n_ties": len(ties)}
        if delta > args.n_sigma * se + args.rho * base:
            termination_reason = "stop_rule"
            stop_index = len(rows) - 1
            rejected = {**extra, "removed": total - nonzero(net),
                        "val_delta": delta, "val_se": se}
            net.load_state_dict(state)
            break
        rows.append(evaluate_row(net, val_b, test, ref_b, total, started, step,
                                 args.device, **extra))
        print(f"berman_validated {step}: removed {rows[-1]['frac_removed']:.1%}, "
              f"val {rows[-1]['val_loss']:.4f}", flush=True)
    save(RESULTS / "berman_validated.json", {
        **COMMON, "status": "complete", "method": "berman_validated",
        "termination_reason": termination_reason,
        "protocol": protocol(args, fisher_subset=args.fisher_n, grid="10%",
                             candidate_pool=args.berman_candidate_pool,
                             candidate_selection="held-out valA loss"),
        "stop_index": stop_index if stop_index is not None else len(rows) - 1,
        "stopping_point": rows[-1], "rejected_candidate": rejected,
        "history": rows, "selections": rows[1:]})


# --------------------------------------------------------------------------
# Fisher geometry.  theta of a layer = [W | b], shape [out, in + 1], flattened row-major.
# --------------------------------------------------------------------------

def theta_of(linear):
    return torch.cat((linear.weight, linear.bias[:, None]), 1).double()


def set_theta(linear, theta):
    linear.weight.copy_(theta[:, :-1])
    linear.bias.copy_(theta[:, -1])


def schur(fisher, theta, deleted):
    remaining = (theta != 0) & (fisher.diagonal(dim1=-2, dim2=-1) > 0) & ~deleted
    block = fisher * remaining[..., :, None] * remaining[..., None, :]
    rhs = (fisher @ (theta * deleted)[..., None])[..., 0] * remaining
    delta = (torch.linalg.pinv(block, hermitian=True) @ rhs[..., None])[..., 0]
    delta -= theta * deleted
    cost = .5 * (delta[..., None, :] @ fisher @ delta[..., None])[..., 0, 0]
    return delta, cost.clamp_min(0)


def schur_invertible(fisher, theta, deleted):
    active = (theta != 0) & (fisher.diagonal() > 0)
    local = fisher[active][:, active]
    eigenvalues, eigenvectors = torch.linalg.eigh((local + local.T) / 2)
    cutoff = eigenvalues[-1].clamp_min(0) * len(eigenvalues) * torch.finfo(
        eigenvalues.dtype).eps
    if eigenvalues[0] <= cutoff:
        raise RuntimeError("singular Fisher block: use schur()")
    inverse = torch.zeros_like(fisher)
    inverse[active[:, None] & active[None, :]] = (
        (eigenvectors / eigenvalues) @ eigenvectors.T).flatten()
    solutions = []
    for mask in deleted:
        on = mask & active
        delta = -inverse[:, on] @ torch.linalg.solve(
            inverse[on][:, on], theta[on])
        solutions.append(torch.where(mask, -theta, delta))
    delta = torch.stack(solutions)
    return delta, (.5 * ((delta @ fisher) * delta).sum(1)).clamp_min(0)


@torch.inference_mode()
def fisher_factors(net, x):
    """Inputs A_l = [a_l, 1] of each linear layer and rows g_l[x, c] = sqrt(p_c)(e_c - p)^T dz/dpre_l.
    F of layer l = mean_x sum_c (g a^T)(g a^T)^T, since sum_c p_c (e_c - p)(e_c - p)^T = diag p - p p^T."""
    l0, l1, l2 = linears(net)
    a0 = x.flatten(1).double()
    h1 = F.linear(a0, l0.weight.double(), l0.bias.double())
    a1 = h1.relu()
    h2 = F.linear(a1, l1.weight.double(), l1.bias.double())
    a2 = h2.relu()
    p = F.linear(a2, l2.weight.double(), l2.bias.double()).softmax(-1)
    g2 = p.sqrt()[:, :, None] * (torch.eye(p.shape[1], dtype=p.dtype, device=p.device)
                                 - p[:, None, :])
    g1 = (g2 @ l2.weight.double()) * (h2 > 0)[:, None]
    g0 = (g1 @ l1.weight.double()) * (h1 > 0)[:, None]
    ones = torch.ones(len(a0), 1, dtype=a0.dtype, device=a0.device)
    return [torch.cat((a, ones), 1) for a in (a0, a1, a2)], [g0, g1, g2]


def singleton_solutions(gram, theta, live):
    """Minimum-norm constrained solutions for deleting each coordinate of small Fisher blocks."""
    masked = gram * live[:, :, None] * live[:, None, :]
    rtol = torch.finfo(torch.float32).eps * max(gram.shape[-1], 1)
    inverse = torch.linalg.pinv(masked, rtol=rtol, hermitian=True)
    projection = (inverse @ masked)
    projection = (projection + projection.transpose(1, 2)) / 2
    null = torch.eye(gram.shape[-1], dtype=gram.dtype,
                     device=gram.device)[None] - projection
    null_diag = null.diagonal(dim1=1, dim2=2)
    inverse_diag = inverse.diagonal(dim1=1, dim2=2)
    zero = null_diag > rtol
    directions = torch.where(
        zero[:, :, None],
        null.transpose(1, 2) / null_diag.clamp_min(rtol)[:, :, None],
        inverse.transpose(1, 2) /
        inverse_diag.clamp_min(torch.finfo(gram.dtype).eps)[:, :, None])
    delta = -theta[:, :, None] * directions
    ids = torch.arange(gram.shape[-1], device=gram.device)
    delta[:, ids, ids] = -theta
    cost = .5 * torch.einsum("okp,opq,okq->ok", delta, gram, delta)
    return delta, cost.clamp_min(0).masked_fill(~live, torch.inf), rtol


def unit_compensations(activations, jacobian, outgoing, bias, units, max_iter=128):
    """Solve all neuron compensation systems with the full Fisher operator, without forming F."""
    x = torch.cat((activations[:, units],
                   torch.ones(len(activations), 1, dtype=activations.dtype,
                              device=activations.device)), 1).double()
    j = jacobian.double()
    v = outgoing[:, units].double()
    candidates, n, classes = len(units), len(x), j.shape[1]
    mask = torch.ones(candidates, v.shape[0], x.shape[1], dtype=torch.bool,
                      device=x.device)
    ids = torch.arange(candidates, device=x.device)
    mask[ids, :, ids] = False
    mask[:, :, :-1] &= (v != 0)[None]
    mask[:, :, -1] &= (bias != 0)[None]
    target = x[:, :-1].T[:, :, None] * torch.einsum("ncr,rk->knc", j, v)

    def operator(value):
        value = value * mask
        preactivation = torch.einsum("nm,krm->knr", x, value)
        effect = torch.einsum("ncr,knr->knc", j, preactivation)
        back = torch.einsum("ncr,knc->knr", j, effect)
        return torch.einsum("knr,nm->krm", back, x) / n

    target_back = torch.einsum("ncr,knc->knr", j, target)
    rhs = torch.einsum("knr,nm->krm", target_back, x) / n
    rhs *= mask
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm0 = residual.flatten(1).square().sum(1).clamp_min(
        torch.finfo(residual.dtype).tiny)
    residual_norm = norm0.clone()
    tolerance = (torch.finfo(torch.float32).eps * max(x.shape[1] * v.shape[0], 1)) ** 2
    iterations = 0
    for iterations in range(1, max_iter + 1):
        image = operator(direction)
        denominator = (direction * image).flatten(1).sum(1)
        active = residual_norm > norm0 * tolerance
        safe = active & (denominator > torch.finfo(denominator.dtype).eps)
        alpha = torch.where(safe, residual_norm / denominator, 0.)
        solution += alpha[:, None, None] * direction
        residual -= alpha[:, None, None] * image
        new_norm = residual.flatten(1).square().sum(1)
        if not bool((new_norm > norm0 * tolerance).any()):
            residual_norm = new_norm
            break
        beta = torch.where(residual_norm > 0, new_norm / residual_norm, 0.)
        direction = residual + beta[:, None, None] * direction
        direction *= mask
        residual_norm = new_norm
    preactivation = torch.einsum("nm,krm->knr", x, solution)
    repaired = torch.einsum("ncr,knr->knc", j, preactivation)
    cost = .5 * (repaired - target).square().sum((1, 2)) / n
    relative_residual = (residual_norm / norm0).sqrt()
    return solution, cost, iterations, relative_residual


def spectral_group_solutions(fisher, theta, deleted):
    active = theta != 0
    local = (fisher[active][:, active] + fisher[active][:, active].T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(local)
    rtol = torch.finfo(torch.float32).eps * local.shape[0]
    keep = eigenvalues > eigenvalues[-1].clamp_min(0) * rtol
    basis = eigenvectors[:, keep]
    inverse_local = (basis / eigenvalues[keep]) @ basis.T
    null_local = torch.eye(local.shape[0], dtype=local.dtype,
                           device=local.device) - basis @ basis.T
    inverse = torch.zeros_like(fisher)
    null = torch.zeros_like(fisher)
    active_block = active[:, None] & active[None, :]
    inverse[active_block] = inverse_local.flatten()
    null[active_block] = null_local.flatten()
    solutions = []
    for deletion in deleted:
        on = deletion & active
        target = -theta[on]
        delta_null = null[:, on] @ torch.linalg.pinv(
            null[on][:, on], rtol=rtol, hermitian=True) @ target
        residual = target - delta_null[on]
        delta_range = inverse[:, on] @ torch.linalg.pinv(
            inverse[on][:, on], rtol=rtol, hermitian=True) @ residual
        delta = delta_null + delta_range
        delta[on] = target
        solutions.append(delta)
    delta = torch.stack(solutions)
    cost = .5 * ((delta @ fisher) * delta).sum(1)
    return delta, cost.clamp_min(0), int(keep.sum())


@torch.inference_mode()
def weight_fisher_candidates(net, A, G, per_family, block_size):
    """One candidate per scalar.  R is its live row block; the row bias is in every weight block."""
    out = []
    for layer, (linear, a, g) in enumerate(zip(linears(net), A, G)):
        q = g.square().sum(1)
        layer_candidates = []
        for start in range(0, linear.in_features, block_size):
            stop = min(start + block_size, linear.in_features)
            features = torch.cat((a[:, start:stop], a[:, -1:]), 1)
            theta = torch.cat((linear.weight[:, start:stop],
                               linear.bias[:, None]), 1).double()
            live = theta != 0
            gram = torch.einsum("xo,xi,xj->oij", q, features, features) / len(a)
            delta, cost, rtol = singleton_solutions(gram, theta, live)
            width = stop - start
            candidate_columns = width
            flat_cost = cost[:, :candidate_columns]
            for flat in torch.argsort(flat_cost.flatten())[:per_family].tolist():
                row, local = divmod(flat, candidate_columns)
                if not torch.isfinite(flat_cost[row, local]):
                    continue
                d = delta[row, local]
                direct = (.5 * torch.dot(d, gram[row] @ d)).clamp_min(0)
                if not torch.isclose(direct, flat_cost[row, local], rtol=1e-8, atol=1e-12):
                    raise RuntimeError("scalar Fisher cost does not match its parameter update")
                column = start + local
                def apply(net, layer=layer, row=row, start=start, stop=stop,
                          column=column, d=d):
                    linear = linears(net)[layer]
                    linear.weight[row, start:stop].add_(d[:stop - start].to(linear.weight))
                    linear.bias[row].add_(d[-1].to(linear.bias))
                    linear.weight[row, column] = 0
                c = float(direct)
                layer_candidates.append({
                    "kind": "weight_fisher",
                    "label": f"weight_fisher:L{layer}:w{row},{column}",
                    "apply": apply, "fisher_cost": c,
                    "fisher_cost_per_removed": c,
                    "fisher_rank": int(torch.linalg.matrix_rank(gram[row], rtol=rtol))})
        out.extend(sorted(layer_candidates,
                          key=lambda candidate: candidate["fisher_cost_per_removed"])
                   [:per_family])
    return out


@torch.inference_mode()
def unit_fisher_candidates(net, A, G, per_family, compensate=True):
    """D = live outgoing weights of one neuron; R = all other live outgoing weights + next bias,
    or R = empty (unit_zero) when compensate is False."""
    out = []
    for layer in (0, 1):
        incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
        a, g = A[layer + 1][:, :-1], G[layer + 1]
        live = outgoing.weight != 0
        units = live.any(0).nonzero().flatten()
        if not units.numel():
            continue
        saved = (live[:, units].sum(0) + (incoming.weight[units] != 0).sum(1)
                 + (incoming.bias[units] != 0)).double()
        if compensate and layer == 0:
            updates, cost, iteration_count, residual = unit_compensations(
                a, g, outgoing.weight, outgoing.bias, units)
            iterations = torch.full((len(units),), iteration_count, dtype=torch.long,
                                    device=units.device)
            tolerance = torch.finfo(torch.float32).eps * max(
                a.shape[1] * outgoing.out_features, 1)
            cost = cost.masked_fill(residual > tolerance, torch.inf)
            rank = -1
        elif compensate:
            features = A[layer + 1]
            theta = theta_of(outgoing).flatten()
            design = (g[:, :, :, None] * features[:, None, None, :]).reshape(
                -1, theta.numel())
            fisher = design.T @ design / len(features)
            deleted = torch.zeros(len(units), theta.numel(), dtype=torch.bool,
                                  device=theta.device)
            stride = features.shape[1]
            rows = torch.arange(outgoing.out_features, device=theta.device)
            for local, unit in enumerate(units):
                deleted[local, rows * stride + unit] = live[:, unit]
            deltas, cost, rank = spectral_group_solutions(fisher, theta, deleted)
            direct = .5 * ((deltas @ fisher) * deltas).sum(1)
            if not torch.allclose(cost, direct.clamp_min(0), rtol=1e-8, atol=1e-10):
                raise RuntimeError("neuron Fisher cost does not match its parameter update")
            residual = torch.zeros_like(cost)
            iterations = torch.zeros_like(cost, dtype=torch.long)
        else:
            contribution = (a[:, units].T[:, :, None]
                            * torch.einsum("ncr,rk->knc", g, outgoing.weight[:, units].double()))
            cost = .5 * contribution.square().sum((1, 2)) / len(a)
            iterations = torch.zeros_like(cost, dtype=torch.long)
            residual = torch.ones_like(cost)
            rank = 0
        score = cost / saved
        for j in score.argsort()[:per_family].tolist():
            if not torch.isfinite(score[j]):
                continue
            i = int(units[j])
            if compensate and layer == 1:
                delta = deltas[j]
                def apply(net, layer=layer, i=i, delta=delta):
                    incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
                    set_theta(outgoing, theta_of(outgoing) + delta.view_as(theta_of(outgoing)))
                    outgoing.weight[:, i] = 0
                    incoming.weight[i] = 0
                    incoming.bias[i] = 0
            else:
                update = updates[j] if compensate else None
                survivors = units
                def apply(net, layer=layer, i=i, update=update, survivors=survivors):
                    incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
                    if update is not None:
                        outgoing.weight[:, survivors] += update[:, :-1].to(outgoing.weight)
                        outgoing.bias.add_(update[:, -1].to(outgoing.bias))
                    outgoing.weight[:, i] = 0
                    incoming.weight[i] = 0
                    incoming.bias[i] = 0
            kind = "unit_fisher" if compensate else "unit_zero"
            out.append({"kind": kind, "label": f"{kind}:L{layer}:n{i}",
                        "apply": apply, "fisher_cost": float(cost[j]),
                        "fisher_cost_per_removed": float(score[j]),
                        "fisher_rank": rank,
                        "fisher_iterations": int(iterations[j]),
                        "fisher_relative_residual": float(residual[j])})
    return out


@torch.inference_mode()
def weight_zero_candidates(net, A, G, per_family):
    """One scalar deletion with no compensation, ranked by the exact Fisher diagonal."""
    out = []
    for layer, (linear, a, g) in enumerate(zip(linears(net), A, G)):
        theta = linear.weight.double()
        diagonal = g.square().sum(1).T @ a[:, :-1].square() / len(a)
        cost = (.5 * theta.square() * diagonal).masked_fill(theta == 0, torch.inf)
        for flat in torch.argsort(cost.flatten())[:per_family].tolist():
            row, column = divmod(flat, linear.in_features)
            def apply(net, layer=layer, row=row, column=column):
                linear = linears(net)[layer]
                linear.weight[row, column] = 0
            c = float(cost[row, column])
            out.append({"kind": "weight_zero",
                        "label": f"weight_zero:L{layer}:w{row},{column}",
                        "apply": apply, "fisher_cost": c,
                        "fisher_cost_per_removed": c})
    return out


@torch.inference_mode()
def fisher_general_candidates(net, A, G, args):
    matrices = [theta_of(linear) for linear in linears(net)]
    sizes = [value.numel() for value in matrices]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    theta = torch.cat([value.flatten() for value in matrices])
    live = theta != 0
    n, classes = len(A[0]), G[0].shape[1]
    diagonal = torch.cat([
        torch.einsum("ncr,nf->rf", g.square(), a.square()).flatten() / n
        for a, g in zip(A, G)])
    closure_cache, target_cache = {}, {}

    def columns(indices):
        indices = torch.as_tensor(indices, device=theta.device, dtype=torch.long)
        result = theta.new_empty(n * classes, len(indices))
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            positions = ((indices >= start) & (indices < stop)).nonzero().flatten()
            if not positions.numel():
                continue
            local = indices[positions] - start
            width = matrices[layer].shape[1]
            rows, cols = torch.div(local, width, rounding_mode="floor"), local % width
            block = G[layer][:, :, rows] * A[layer][:, None, cols]
            result[:, positions] = block.reshape(n * classes, -1) / math.sqrt(n)
        return result

    def closure(deleted):
        deleted = tuple(sorted(set(deleted)))
        if deleted not in closure_cache:
            values = [matrix.clone() for matrix in matrices]
            ids = torch.tensor(deleted, device=theta.device)
            for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
                local = ids[(ids >= start) & (ids < stop)] - start
                if local.numel():
                    values[layer].view(-1)[local] = 0
            canonicalize_mlp_tensors(
                [value[:, :-1] for value in values],
                [value[:, -1] for value in values], (F.relu, F.relu))
            survivor = torch.cat([(value != 0).flatten() for value in values])
            closure_cache[deleted] = survivor, int(live.sum() - survivor.sum())
        return closure_cache[deleted]

    def fit(deleted, updated=()):
        deleted = tuple(sorted(set(deleted)))
        if not deleted:
            return None
        survivor, savings = closure(deleted)
        updated = tuple(sorted(i for i in set(updated)
                               if bool(survivor[i]) and i not in deleted))
        if deleted not in target_cache:
            ids = torch.tensor(deleted, device=theta.device)
            target_cache[deleted] = columns(ids) @ theta[ids]
        target = target_cache[deleted]
        if updated:
            z = columns(updated)
            matrix = z.T @ z
            if args.fisher_general_mu:
                matrix.diagonal().add_(args.fisher_general_mu)
            rhs = z.T @ target
            factor, info = torch.linalg.cholesky_ex(matrix)
            if int(info) == 0:
                delta = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
            else:
                rtol = torch.finfo(torch.float32).eps * len(updated)
                delta = torch.linalg.pinv(matrix, rtol=rtol, hermitian=True) @ rhs
            residual = z @ delta - target
        else:
            delta = target.new_empty(0)
            residual = -target
        damage = .5 * residual.square().sum()
        objective = (damage + args.fisher_general_lambda_u * len(updated)
                     + .5 * args.fisher_general_mu * delta.square().sum()
                     - args.fisher_general_lambda_d * savings)
        return {"deleted": deleted, "updated": updated, "delta": delta,
                "damage": float(damage), "objective": float(objective),
                "savings": savings, "residual": residual}

    weight_ids = []
    for layer, linear in enumerate(linears(net)):
        width = linear.in_features + 1
        row = torch.arange(linear.out_features, device=theta.device)[:, None]
        col = torch.arange(linear.in_features, device=theta.device)[None]
        ids = offsets[layer] + row * width + col
        weight_ids.append(ids[linear.weight != 0])
    weight_ids = torch.cat(weight_ids)
    scalar_cost = .5 * theta[weight_ids].square() * diagonal[weight_ids]
    scalar_ids = weight_ids[torch.argsort(scalar_cost)]
    atoms = [tuple(chunk.tolist()) for chunk in scalar_ids.split(
        args.fisher_general_window)[:args.fisher_general_pool] if chunk.numel()]
    atoms.extend((int(index),) for index in scalar_ids[:args.fisher_general_pool])

    groups = []
    for layer in (0, 1):
        outgoing = linears(net)[layer + 1]
        stride = outgoing.in_features + 1
        for unit in (outgoing.weight != 0).any(0).nonzero().flatten().tolist():
            rows = (outgoing.weight[:, unit] != 0).nonzero().flatten()
            group = tuple((offsets[layer + 1] + rows * stride + unit).tolist())
            candidate = fit(group)
            groups.append((candidate["objective"] / max(candidate["savings"], 1), group))
    groups.sort(key=lambda item: item[0])
    atoms.extend(group for _, group in groups[:args.fisher_general_pool])
    atoms = list(dict.fromkeys(atoms))
    initial_fits = [fit(atom) for atom in atoms]
    initial_fits = [item for item in initial_fits
                    if item["savings"] >= args.fisher_general_min_savings]
    starts = sorted(initial_fits, key=lambda item: item["objective"])
    starts = starts[:args.fisher_general_starts]

    def correlations(residual):
        shaped = residual.reshape(n, classes)
        return torch.cat([
            (torch.einsum("ncr,nc,nf->rf", g, shaped, a) / math.sqrt(n)).flatten()
            for a, g in zip(A, G)])

    found = {(item["deleted"], item["updated"]): item for item in initial_fits}
    for initial in starts:
        current = initial
        for _ in range(args.fisher_general_iters):
            alternatives = []
            for atom in atoms:
                deletion = tuple(sorted(set(current["deleted"]).union(atom)))
                if deletion != current["deleted"]:
                    alternatives.append(fit(deletion, current["updated"]))
            restore = sorted(current["deleted"],
                             key=lambda index: float(theta[index].square() * diagonal[index]),
                             reverse=True)[:args.fisher_general_update_pool]
            alternatives.extend(
                fit(tuple(index for index in current["deleted"] if index != restored),
                    current["updated"])
                for restored in restore if len(current["deleted"]) > 1)
            survivor, _ = closure(current["deleted"])
            available = survivor.clone()
            available[torch.tensor(current["deleted"], device=theta.device)] = False
            if current["updated"]:
                available[torch.tensor(current["updated"], device=theta.device)] = False
            available_ids = available.nonzero().flatten()
            if available_ids.numel():
                score = correlations(current["residual"])[available_ids].abs()
                scale = (diagonal[available_ids] + args.fisher_general_mu).sqrt().clamp_min(1e-15)
                additions = available_ids[torch.argsort(score / scale, descending=True)
                                          [:args.fisher_general_update_pool]]
                alternatives.extend(fit(current["deleted"],
                                        current["updated"] + (int(index),))
                                    for index in additions)
            alternatives.extend(fit(current["deleted"],
                                    tuple(index for index in current["updated"]
                                          if index != removed))
                                for removed in current["updated"])
            alternatives = [candidate for candidate in alternatives if candidate is not None]
            if not alternatives:
                break
            best = min(alternatives, key=lambda item: item["objective"])
            if best["objective"] >= current["objective"] - 1e-14:
                break
            current = best
            found[(current["deleted"], current["updated"])] = current
        found[(current["deleted"], current["updated"])] = current

    proposals = []
    for candidate in sorted(found.values(), key=lambda item: item["objective"]):
        if (candidate["objective"] >= 0
                or candidate["savings"] < args.fisher_general_min_savings):
            continue
        deleted = torch.tensor(candidate["deleted"], device=theta.device)
        updated = torch.tensor(candidate["updated"], device=theta.device)
        delta = candidate["delta"]

        def apply(net, deleted=deleted, updated=updated, delta=delta):
            values = torch.cat([theta_of(linear).flatten() for linear in linears(net)])
            if updated.numel():
                values[updated] += delta
            values[deleted] = 0
            for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
                set_theta(linears(net)[layer], values[start:stop].reshape_as(matrices[layer]))

        proposals.append({
            "kind": "fisher_general",
            "label": (f"fisher_general:D{len(deleted)}:U{len(updated)}:"
                      f"K{candidate['savings']}"),
            "apply": apply, "fisher_cost": candidate["damage"],
            "fisher_cost_per_removed": candidate["damage"] / candidate["savings"],
            "search_objective": candidate["objective"],
            "explicit_deletions": len(deleted), "compensating_updates": len(updated),
            "predicted_savings": candidate["savings"]})
    return proposals


# --------------------------------------------------------------------------
# Classic unit moves (activation statistics on training data)
# --------------------------------------------------------------------------

@torch.inference_mode()
def unit_classic_candidates(net, x, per_family):
    l0, l1, _ = linears(net)
    a1 = F.relu(l0(x.flatten(1)))
    acts = (a1, F.relu(l1(a1)))
    out = []
    for layer, a in enumerate(acts):
        incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
        alive = ((incoming.weight != 0).any(1) | (incoming.bias != 0)) & (outgoing.weight != 0).any(0)
        alive = alive.nonzero().flatten()
        if not alive.numel():
            continue
        a = a[:, alive]
        norms = outgoing.weight[:, alive].square().sum(0)
        means = a.mean(0)
        centered = a - means
        variance = centered.square().mean(0)
        for kind, score in (("unit_off", a.square().mean(0) * norms),
                            ("unit_mean", variance * norms)):
            for local in torch.argsort(score)[:per_family].tolist():
                c = float(means[local]) if kind == "unit_mean" else 0.
                out.append(unit_move(kind, layer, int(alive[local]), c=c))
        if alive.numel() > 1:
            covariance = centered.T @ centered / len(a)
            alpha = covariance / variance.clamp_min(1e-12)[None, :]
            beta = means[:, None] - alpha * means[None, :]
            residual = variance[:, None] - covariance.square() / variance.clamp_min(1e-12)[None, :]
            cost = residual.clamp_min(0) * norms[:, None]
            cost.fill_diagonal_(torch.inf)
            for flat in torch.argsort(cost.flatten())[:per_family].tolist():
                s, t = divmod(flat, alive.numel())
                out.append(unit_move("unit_merge", layer, int(alive[s]), int(alive[t]),
                                     float(alpha[s, t]), float(beta[s, t])))
    return out


def unit_move(kind, layer, i, target=-1, alpha=0., c=0.):
    """unit_off: a_i -> 0.  unit_mean: a_i -> c.  unit_merge: a_i -> alpha a_target + c."""
    def apply(net):
        incoming, outgoing = linears(net)[layer], linears(net)[layer + 1]
        column = outgoing.weight[:, i].clone()
        outgoing.bias.add_(c * column)
        if target >= 0:
            outgoing.weight[:, target].add_(alpha * column)
        incoming.weight[i] = 0
        incoming.bias[i] = 0
        outgoing.weight[:, i] = 0
    label = f"{kind}:L{layer}:n{i}" + (f"->{target}" if target >= 0 else "")
    return {"kind": kind, "label": label, "apply": apply}


# --------------------------------------------------------------------------
# Greedy path shared by classic and Fisher
# --------------------------------------------------------------------------

@torch.inference_mode()
def canonicalize(net):
    canonicalize_linear_chain(linears(net), (F.relu, F.relu))


def snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


MOVES = {"classic": ["unit_off", "unit_mean", "unit_merge"],
         "classic_weights": ["unit_off", "unit_mean", "unit_merge", "weight_zero"],
         "fisher": ["unit_fisher", "weight_fisher"],
         "fisher_original": ["unit_fisher", "weight_fisher"],
         "fisher_general": ["fisher_general"],
         "fisher_general_score": ["fisher_general"]}


def run_greedy(args, method, train, val_a, val_b, test):
    net = model(args.device)
    total = sum(p.numel() for p in net.parameters())
    ref_a, ref_b = losses(net, val_a, args.device), losses(net, val_b, args.device)
    base = float(ref_b.mean())
    x_fisher = fisher_images(train, args.fisher_n)[0].to(args.device)
    x_stats = train[0][:4096].to(args.device)
    original_factors = fisher_factors(net, x_fisher) if method == "fisher_original" else None
    started = time.perf_counter()
    rows = [evaluate_row(net, val_b, test, ref_b, total, started, 0, args.device)]
    stop_index, rejected, termination_reason = None, None, "max_steps"
    name = method
    record = lambda status: save(RESULTS / f"{name}.json", {
        **COMMON, "status": status, "method": method,
        "termination_reason": termination_reason,
        "protocol": protocol(args, fisher_subset=args.fisher_n,
                             per_family=args.per_family,
                             fisher_weight_block=args.fisher_weight_block,
                             fisher_type="exact class-marginal model Fisher",
                             fisher_reference=("original checkpoint" if original_factors
                                               is not None else "current model"),
                             scalar_compensation="same output-row block plus bias",
                             neuron_compensation="all surviving outgoing weights plus bias",
                             candidate_ranking="Fisher cost per parameter removed",
                             candidate_selection=("search objective" if method ==
                                                  "fisher_general_score" else
                                                  "held-out valA loss"),
                             fisher_general={
                                 "lambda_u": args.fisher_general_lambda_u,
                                 "lambda_d": args.fisher_general_lambda_d,
                                 "mu": args.fisher_general_mu,
                                 "window": args.fisher_general_window,
                                 "min_savings": args.fisher_general_min_savings,
                                 "pool": args.fisher_general_pool,
                                 "starts": args.fisher_general_starts,
                                 "iterations": args.fisher_general_iters,
                                 "update_pool": args.fisher_general_update_pool},
                             moves=MOVES[method]),
        "stop_index": stop_index if stop_index is not None else len(rows) - 1,
        "stopping_point": rows[-1], "rejected_candidate": rejected, "history": rows,
        "selections": rows[1:],
        "move_counts": {k: sum(r.get("kind") == k for r in rows)
                        for k in sorted({r["kind"] for r in rows[1:]})}})
    for step in range(1, args.max_steps + 1):
        needs_fisher = bool({"unit_fisher", "weight_fisher", "weight_zero",
                             "fisher_general"} & set(MOVES[method]))
        A, G = (original_factors or fisher_factors(net, x_fisher)) if needs_fisher else (None, None)
        proposals = []
        if "unit_fisher" in MOVES[method]:
            proposals += unit_fisher_candidates(net, A, G, args.per_family)
        if "weight_fisher" in MOVES[method]:
            proposals += weight_fisher_candidates(
                net, A, G, args.per_family, args.fisher_weight_block)
        if "fisher_general" in MOVES[method]:
            proposals += fisher_general_candidates(net, A, G, args)
        if "unit_mean" in MOVES[method]:
            proposals += unit_classic_candidates(net, x_stats, args.per_family)
        if "weight_zero" in MOVES[method]:
            proposals += weight_zero_candidates(net, A, G, args.per_family)
        score_only = method == "fisher_general_score"
        before, state, evaluated = nonzero(net), snapshot(net), []
        for proposal in proposals:
            proposal["apply"](net)
            canonicalize(net)
            saved = before - nonzero(net)
            if saved > 0:
                candidate_loss = None if score_only else losses(net, val_a, args.device)
                evaluated.append((proposal, candidate_loss, saved))
            net.load_state_dict(state)
        if not evaluated:
            termination_reason = "candidate_exhaustion"
            break
        if score_only:
            chosen, _unused_loss, saved = min(
                evaluated, key=lambda item: (item[0]["search_objective"], -item[2]))
            ties = [(chosen, None, saved)]
        else:
            best = min(evaluated, key=lambda item: float(item[1].mean()))[1]
            ties = [item for item in evaluated
                    if paired(item[1], best)[0] <= args.n_sigma * paired(item[1], best)[1]]
            chosen, chosen_loss, saved = min(
                ties, key=lambda item: (-item[2], float(item[1].mean())))
        chosen["apply"](net)
        canonicalize(net)
        chosen_loss = losses(net, val_a, args.device)
        delta, se = paired(losses(net, val_b, args.device), ref_b)
        extra = {"kind": chosen["kind"], "move": chosen["label"], "actual_savings": saved,
                 "val_a_loss": float(chosen_loss.mean()), "n_candidates": len(evaluated),
                 "n_ties": len(ties)}
        for key in ("fisher_cost", "fisher_cost_per_removed", "fisher_rank",
                    "fisher_iterations", "fisher_relative_residual", "search_objective",
                    "explicit_deletions", "compensating_updates", "predicted_savings"):
            if key in chosen:
                extra[key] = chosen[key]
        if delta > args.n_sigma * se + args.rho * base:
            termination_reason = "stop_rule"
            stop_index = len(rows) - 1
            rejected = {**extra, "removed": total - nonzero(net), "val_delta": delta, "val_se": se}
            net.load_state_dict(state)
            break
        rows.append(evaluate_row(net, val_b, test, ref_b, total, started, step, args.device,
                                 **extra))
        record("running")
        print(f"{method} {step}: {chosen['label']} saved {saved}, removed "
              f"{rows[-1]['frac_removed']:.1%}, val {rows[-1]['val_loss']:.4f}, "
              f"{rows[-1]['seconds']:.0f}s", flush=True)
    record("complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("berman", "berman_validated", *MOVES, "all"),
                        default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--fisher-n", type=int, default=500)
    parser.add_argument("--fisher-weight-block", type=int, default=64)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--berman-candidate-pool", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fisher-general-lambda-u", type=float, default=1e-8)
    parser.add_argument("--fisher-general-lambda-d", type=float, default=1e-7)
    parser.add_argument("--fisher-general-mu", type=float, default=1e-4)
    parser.add_argument("--fisher-general-window", type=int, default=256)
    parser.add_argument("--fisher-general-min-savings", type=int, default=32)
    parser.add_argument("--fisher-general-pool", type=int, default=8)
    parser.add_argument("--fisher-general-starts", type=int, default=2)
    parser.add_argument("--fisher-general-iters", type=int, default=4)
    parser.add_argument("--fisher-general-update-pool", type=int, default=4)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    train, test_data = mnist()
    order = torch.randperm(len(test_data[0]), generator=torch.Generator().manual_seed(SPLIT_SEED))
    split = lambda index: (test_data[0][order[index]], test_data[1][order[index]])
    val_a, val_b, test = (loader(split(slice(0, 2500)), 512), loader(split(slice(2500, 5000)), 512),
                          loader(split(slice(5000, None)), 512))
    if args.method in ("berman", "all"):
        run_berman(args, train, val_b, test, args.fisher_n)
    if args.method in ("berman_validated", "all"):
        run_berman_validated(args, train, val_a, val_b, test)
    for method in MOVES:
        if args.method in (method, "all"):
            run_greedy(args, method, train, val_a, val_b, test)


if __name__ == "__main__":
    main()
