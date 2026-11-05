#!/usr/bin/env python3
"""Berman, Classic, and Fisher General on SimpleNN/MNIST."""
from __future__ import annotations

import argparse
import gzip
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
from nnprune.factorized_fisher import inverse_diagonal
from native_baselines.result_artifacts import write_json as save
from native_baselines.sparsegpt import sparsegpt_matrix

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


def theta_of(linear):
    return torch.cat((linear.weight, linear.bias[:, None]), 1).double()


def set_theta(linear, theta):
    linear.weight.copy_(theta[:, :-1])
    linear.bias.copy_(theta[:, -1])


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


@torch.inference_mode()
def run_fisher_sparsegpt(args, train, val_b, test):
    started = time.perf_counter()
    reference_net = model(args.device)
    reference = losses(reference_net, val_b, args.device)
    total = sum(p.numel() for p in reference_net.parameters())
    calibration = loader(fisher_images(train, args.fisher_n), 128)
    factors = [torch.zeros((layer.out_features, layer.out_features),
                           device=args.device, dtype=torch.float64)
               for layer in linears(reference_net)]
    observations = 0
    for images, _ in calibration:
        _, gradients = fisher_factors(reference_net, images.to(args.device))
        for factor, gradient in zip(factors, gradients):
            factor.add_(torch.einsum("nci,ncj->ij", gradient, gradient))
        observations += len(images)
    for factor in factors:
        factor.div_(observations)
        factor.diagonal().add_(args.sparsegpt_b_damp * max(
            float(factor.diag().mean()), torch.finfo(factor.dtype).eps))
    inverse_diagonals = [inverse_diagonal(factor) for factor in factors]
    rows = [evaluate_row(reference_net, val_b, test, reference, total, started, 0, args.device)]
    module_records, rejected = [], None
    termination = "targets_exhausted"
    destination = RESULTS / "fisher_sparsegpt.json"

    def record(status):
        save(destination, {
            **COMMON, "status": status, "method": "fisher_sparsegpt",
            "termination_reason": termination,
            "protocol": protocol(
                args, fisher_subset=args.fisher_n,
                fisher_type="class-marginal AXB K-FAC",
                B="frozen original-model class-marginal downstream factor",
                A="current upstream activations, refreshed before each linear module",
                compensation="SparseGPT later-column propagation; generalized OBS AXB ranking",
                biases="fixed; post hoc canonicalization may remove redundant parameters",
                targets=args.sparsegpt_targets, percdamp=args.sparsegpt_percdamp,
                b_damp_ratio=args.sparsegpt_b_damp, blocksize=args.sparsegpt_blocksize),
            "stop_index": len(rows) - 1, "stopping_point": rows[-1],
            "rejected_candidate": rejected, "history": rows,
            "module_records": module_records})

    record("running")
    for step, target in enumerate(args.sparsegpt_targets, 1):
        net = model(args.device)
        modules = []
        for index, layer in enumerate(linears(net)):
            a = torch.zeros((layer.in_features, layer.in_features),
                            dtype=torch.float64, device=args.device)
            for images, _ in calibration:
                activation = images.to(args.device).flatten(1)
                for upstream in linears(net)[:index]:
                    activation = F.relu(upstream(activation))
                a.add_(activation.double().T @ activation.double())
            a.div_(observations)
            before = layer.weight.detach().clone()
            pruned, mask = sparsegpt_matrix(
                before, a, target, args.sparsegpt_percdamp, args.sparsegpt_blocksize,
                output_inverse_diagonal=inverse_diagonals[index])
            layer.weight.copy_(pruned)
            delta = (pruned - before).double()
            modules.append({"layer": index, "masked_weights": int(mask.sum()),
                            "predicted_fisher_damage": float(
                                .5 * (delta * (factors[index] @ delta @ a)).sum())})
        canonicalize(net)
        row = evaluate_row(net, val_b, test, reference, total, started, step, args.device,
                           kind="fisher_sparsegpt", target_sparsity=target)
        admissible = row["val_delta"] <= args.n_sigma * row["val_se"] + args.rho * float(reference.mean())
        row["admissible"] = admissible
        module_records.append({"target": target, "modules": modules})
        print(f"Fisher-aware SparseGPT {target:.0%}: removed {row['frac_removed']:.2%}, "
              f"validation loss {row['val_loss']:.6f}, admissible={admissible}, "
              f"elapsed {row['seconds']:.1f}s", flush=True)
        if not admissible:
            rejected = {**row, "relative_damage": row["val_delta"] / float(reference.mean())}
            termination = "stop_rule"
            break
        rows.append(row)
        record("running")
    record("complete")


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
def _logic_and(model, terms, name):
    if any(isinstance(term, int) and term == 0 for term in terms):
        return 0
    variables = [term for term in terms if not isinstance(term, int)]
    if not variables:
        return 1
    if len(variables) == 1:
        return variables[0]
    result = model.addVar(vtype="B", name=name)
    model.addGenConstrAnd(result, variables)
    return result


def _logic_or(model, terms, name):
    if any(isinstance(term, int) and term == 1 for term in terms):
        return 1
    variables = [term for term in terms if not isinstance(term, int)]
    if not variables:
        return 0
    if len(variables) == 1:
        return variables[0]
    result = model.addVar(vtype="B", name=name)
    model.addGenConstrOr(result, variables)
    return result


@torch.inference_mode()
def fisher_general_candidates(net, A, G, args):
    try:
        import gurobipy as gp
    except ImportError as error:
        raise RuntimeError("Fisher General requires gurobipy") from error

    matrices = [theta_of(linear) for linear in linears(net)]
    offsets = [0]
    for matrix in matrices:
        offsets.append(offsets[-1] + matrix.numel())
    theta = torch.cat([matrix.flatten() for matrix in matrices])
    live = theta != 0
    n, classes = len(A[0]), G[0].shape[1]

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
            result[:, positions] = (
                G[layer][:, :, rows] * A[layer][:, None, cols]
            ).reshape(n * classes, -1) / math.sqrt(n)
        return result

    def closure_details(deleted):
        values = [matrix.clone() for matrix in matrices]
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            local = torch.tensor([index - start for index in deleted
                                  if start <= index < stop], device=theta.device)
            if local.numel():
                values[layer].view(-1)[local] = 0
        canonicalize_mlp_tensors([value[:, :-1] for value in values],
                                 [value[:, -1] for value in values],
                                 (F.relu, F.relu))
        survivor = torch.cat([(value != 0).flatten() for value in values])
        removed = set((live & ~survivor).nonzero().flatten().tolist())
        return len(removed), removed

    def closure(deleted):
        return closure_details(deleted)[0]

    diagonal = torch.cat([
        torch.einsum("ncr,nf->rf", g.square(), a.square()).flatten() / n
        for a, g in zip(A, G)])
    weight_ids = []
    for layer, linear in enumerate(linears(net)):
        width = linear.in_features + 1
        rows = torch.arange(linear.out_features, device=theta.device)[:, None]
        cols = torch.arange(linear.in_features, device=theta.device)[None]
        ids = offsets[layer] + rows * width + cols
        weight_ids.append(ids[linear.weight != 0])
    weight_ids = torch.cat(weight_ids)
    order = weight_ids[torch.argsort(.5 * theta[weight_ids].square() * diagonal[weight_ids])]

    groups = []
    for layer in (0, 1):
        outgoing = linears(net)[layer + 1]
        width = outgoing.in_features + 1
        for unit in (outgoing.weight != 0).any(0).nonzero().flatten().tolist():
            rows = (outgoing.weight[:, unit] != 0).nonzero().flatten()
            group = tuple((offsets[layer + 1] + rows * width + unit).tolist())
            if len(group) > args.fisher_general_max_group:
                continue
            raw = float((.5 * theta[list(group)].square() * diagonal[list(group)]).sum())
            groups.append((raw / max(closure(group), 1), group))
    groups.sort(key=lambda item: item[0])
    structural_blocks = [group for _, group in groups[:args.fisher_general_group_pool]]
    structural_closures = [closure_details(group)[1] for group in structural_blocks]
    reserved = (set().union(*structural_closures) if structural_closures else set())
    guards = set()
    for layer in (0, 1):
        incoming = linears(net)[layer]
        incoming_width = incoming.in_features + 1
        for unit in (incoming.weight != 0).any(1).nonzero().flatten().tolist():
            ids = [offsets[layer] + unit * incoming_width + source
                   for source in (incoming.weight[unit] != 0).nonzero().flatten().tolist()]
            available = [index for index in ids if index not in reserved]
            if available:
                guards.add(available[-1])
        outgoing = linears(net)[layer + 1]
        width = outgoing.in_features + 1
        for unit in (outgoing.weight != 0).any(0).nonzero().flatten().tolist():
            ids = [offsets[layer + 1] + row * width + unit
                   for row in (outgoing.weight[:, unit] != 0).nonzero().flatten().tolist()]
            available = [index for index in ids if index not in reserved]
            if available:
                guards.add(available[-1])
    scalar_ids = [int(index) for index in order
                  if int(index) not in reserved and int(index) not in guards]
    scalar_blocks = [tuple(block) for block in
                     (scalar_ids[start:start + args.fisher_general_window]
                      for start in range(0, len(scalar_ids), args.fisher_general_window))
                     if block][:args.fisher_general_pool]
    blocks = scalar_blocks + structural_blocks
    if not blocks:
        return []
    block_savings = [len(block) for block in scalar_blocks]
    block_savings += [closure(block) for block in structural_blocks]
    deletion_pool = set().union(*map(set, blocks))

    target = sum((columns(block) @ theta[list(block)] for block in blocks),
                 theta.new_zeros(n * classes))
    live_ids = live.nonzero().flatten()
    scale = (diagonal + args.fisher_general_mu).sqrt().clamp_min(1e-15)
    score = (columns(live_ids).T @ target).abs()
    ranked_updates = live_ids[torch.argsort(score / scale[live_ids], descending=True)]
    update_ids = tuple(int(index) for index in ranked_updates
                       if int(index) not in deletion_pool and int(index) not in reserved
                       )[:args.fisher_general_pool]
    block_effects = torch.stack([-(columns(block) @ theta[list(block)])
                                 for block in blocks], 1)
    update_design = columns(update_ids)
    design = torch.cat((block_effects, update_design), 1)
    q = design.T @ design
    q = (q + q.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(q)
    q = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T

    solver = gp.Model("fisher_general")
    solver.Params.OutputFlag = int(args.fisher_general_log)
    solver.Params.TimeLimit = args.fisher_general_time_limit
    solver.Params.MIPGap = args.fisher_general_gap
    solver.Params.NonConvex = 0
    selected = [solver.addVar(vtype=gp.GRB.BINARY, name=f"block_{index}")
                for index in range(len(blocks))]

    median = float(theta[live].abs().median())
    u, c, bounds = [], [], []
    for index in update_ids:
        uv = solver.addVar(vtype=gp.GRB.BINARY, name=f"u_{index}")
        bound = args.fisher_general_bound * max(abs(float(theta[index])), median, 1e-6)
        cv = solver.addVar(lb=-bound, ub=bound, name=f"c_{index}")
        solver.addGenConstrIndicator(uv, 0, cv == 0.)
        u.append(uv)
        c.append(cv)
        bounds.append(bound)

    savings = gp.quicksum(value * reward
                              for value, reward in zip(selected, block_savings))

    objective_scale = max(float(q.diag().max()), args.fisher_general_mu,
                          args.fisher_general_lambda_u,
                          args.fisher_general_lambda_d, 1e-12)
    variables = selected + c
    objective = gp.QuadExpr()
    for row in range(len(variables)):
        for column in range(len(variables)):
            coefficient = .5 * float(q[row, column]) / objective_scale
            if coefficient:
                objective += coefficient * variables[row] * variables[column]
    for uv, cv in zip(u, c):
        objective += (.5 * args.fisher_general_mu / objective_scale) * cv * cv
        objective += (args.fisher_general_lambda_u / objective_scale) * uv
    objective -= (args.fisher_general_lambda_d / objective_scale) * savings
    solver.setObjective(objective, gp.GRB.MINIMIZE)
    solver.addConstr(gp.quicksum(selected) >= 1)

    for index, variable in enumerate(selected):
        variable.Start = int(index == 0)
    for variable, compensation in zip(u, c):
        variable.Start = 0
        compensation.Start = 0.
    solver.update()
    if args.fisher_general_log:
        print(f"MIQP size: {solver.NumVars} variables, {solver.NumConstrs} constraints, "
              f"{solver.NumNZs} nonzeros", flush=True)
    solver.optimize()
    if solver.SolCount == 0:
        status = solver.Status
        solver.dispose()
        raise RuntimeError(f"Gurobi found no feasible solution (status {status})")

    chosen_blocks = tuple(index for index, variable in enumerate(selected) if variable.X > .5)
    deleted = tuple(sorted(set().union(*(set(blocks[index]) for index in chosen_blocks))))
    updated = tuple(index for position, index in enumerate(update_ids) if u[position].X > .5)
    compensation = torch.tensor([c[update_ids.index(index)].X for index in updated],
                                dtype=theta.dtype, device=theta.device)
    effect = block_effects[:, list(chosen_blocks)].sum(1)
    if updated:
        effect += columns(updated) @ compensation
    damage = float(.5 * effect.square().sum())
    actual_savings = closure(deleted)
    direct_objective = (damage + .5 * args.fisher_general_mu * float(compensation.square().sum())
                        + args.fisher_general_lambda_u * len(updated)
                        - args.fisher_general_lambda_d * actual_savings)
    solver_objective = solver.ObjVal * objective_scale
    modeled_savings = round(float(savings.getValue()))
    if modeled_savings != actual_savings:
        solver.dispose()
        raise RuntimeError("Gurobi canonicalization reward disagrees with the canonicalizer: "
                           f"modeled {modeled_savings}, actual {actual_savings}, "
                           f"blocks {chosen_blocks}")
    tolerance = 1e-7 * max(1., abs(direct_objective), abs(solver_objective))
    if abs(direct_objective - solver_objective) > tolerance:
        solver.dispose()
        raise RuntimeError("Gurobi objective disagrees with direct Fisher evaluation")

    gap = float(solver.MIPGap)
    runtime = float(solver.Runtime)
    status = int(solver.Status)
    bound_active = any(abs(c[update_ids.index(index)].X) >=
                       .999 * bounds[update_ids.index(index)] for index in updated)
    solver.dispose()
    deleted_tensor = torch.tensor(deleted, device=theta.device)
    updated_tensor = torch.tensor(updated, device=theta.device)

    def apply(net):
        values = torch.cat([theta_of(linear).flatten() for linear in linears(net)])
        if updated_tensor.numel():
            values[updated_tensor] += compensation
        values[deleted_tensor] = 0
        for layer, (start, stop) in enumerate(zip(offsets, offsets[1:])):
            set_theta(linears(net)[layer], values[start:stop].reshape_as(matrices[layer]))

    return [{
        "kind": "fisher_general",
        "label": (f"fisher_general:B{len(chosen_blocks)}:D{len(deleted)}:"
                  f"U{len(updated)}:K{actual_savings}"),
        "apply": apply, "fisher_cost": damage,
        "fisher_cost_per_removed": damage / actual_savings,
        "search_objective": direct_objective,
        "explicit_deletions": len(deleted), "compensating_updates": len(updated),
        "predicted_savings": actual_savings, "solver_objective": solver_objective,
        "solver_status": status, "solver_gap": gap, "solver_seconds": runtime,
        "compensation_bound_active": bound_active,
        "candidate_blocks": len(blocks), "candidate_deletions": len(deletion_pool),
        "candidate_updates": len(update_ids),
    }]


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
# Greedy path shared by Classic and Fisher General
# --------------------------------------------------------------------------

@torch.inference_mode()
def canonicalize(net):
    canonicalize_linear_chain(linears(net), (F.relu, F.relu))


def snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


MOVES = {"classic": ("unit_off", "unit_mean", "unit_merge"),
         "classic_weights": ("unit_off", "unit_mean", "unit_merge", "weight_zero"),
         "fisher_general": ("fisher_general",)}


def run_greedy(args, method, train, val_a, val_b, test):
    net = model(args.device)
    total = sum(p.numel() for p in net.parameters())
    ref_a, ref_b = losses(net, val_a, args.device), losses(net, val_b, args.device)
    base = float(ref_b.mean())
    x_fisher = fisher_images(train, args.fisher_n)[0].to(args.device)
    x_stats = train[0][:4096].to(args.device)
    started = time.perf_counter()
    rows = [evaluate_row(net, val_b, test, ref_b, total, started, 0, args.device)]
    stop_index, rejected, termination_reason = None, None, "max_steps"
    name = method
    record = lambda status: save(RESULTS / f"{name}.json", {
        **COMMON, "status": status, "method": method,
        "termination_reason": termination_reason,
        "protocol": protocol(args, fisher_subset=args.fisher_n,
                             per_family=args.per_family,
                             fisher_type="exact class-marginal model Fisher",
                             fisher_reference="current model",
                             candidate_ranking="Fisher cost per parameter removed",
                             candidate_selection="held-out valA loss",
                             fisher_general={
                                 "lambda_u": args.fisher_general_lambda_u,
                                 "lambda_d": args.fisher_general_lambda_d,
                                 "mu": args.fisher_general_mu,
                                 "pool": args.fisher_general_pool,
                                 "window": args.fisher_general_window,
                                 "group_pool": args.fisher_general_group_pool,
                                 "max_group": args.fisher_general_max_group,
                                 "time_limit": args.fisher_general_time_limit,
                                 "gap": args.fisher_general_gap,
                                 "bound": args.fisher_general_bound},
                             moves=MOVES[method]),
        "stop_index": stop_index if stop_index is not None else len(rows) - 1,
        "stopping_point": rows[-1], "rejected_candidate": rejected, "history": rows,
        "selections": rows[1:],
        "move_counts": {k: sum(r.get("kind") == k for r in rows)
                        for k in sorted({r["kind"] for r in rows[1:]})}})
    for step in range(1, args.max_steps + 1):
        needs_fisher = bool({"weight_zero", "fisher_general"} & set(MOVES[method]))
        A, G = fisher_factors(net, x_fisher) if needs_fisher else (None, None)
        proposals = []
        if "fisher_general" in MOVES[method]:
            proposals += fisher_general_candidates(net, A, G, args)
        if "unit_mean" in MOVES[method]:
            proposals += unit_classic_candidates(net, x_stats, args.per_family)
        if "weight_zero" in MOVES[method]:
            proposals += weight_zero_candidates(net, A, G, args.per_family)
        before, state, evaluated = nonzero(net), snapshot(net), []
        for proposal in proposals:
            proposal["apply"](net)
            canonicalize(net)
            saved = before - nonzero(net)
            if saved > 0:
                evaluated.append((proposal, losses(net, val_a, args.device), saved))
            net.load_state_dict(state)
        if not evaluated:
            termination_reason = "candidate_exhaustion"
            break
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
        for key in ("fisher_cost", "fisher_cost_per_removed", "search_objective",
                    "explicit_deletions", "compensating_updates", "predicted_savings",
                    "solver_objective", "solver_status", "solver_gap", "solver_seconds",
                    "compensation_bound_active", "candidate_deletions",
                    "candidate_updates", "candidate_blocks"):
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


def main(argv=None):
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("berman", "fisher_sparsegpt", *MOVES, "all"),
                        default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--fisher-n", type=int, default=500)
    parser.add_argument("--sparsegpt-targets", default=".1,.2,.3,.4,.5,.6,.7,.8,.9")
    parser.add_argument("--sparsegpt-blocksize", type=int, default=128)
    parser.add_argument("--sparsegpt-percdamp", type=float, default=.01)
    parser.add_argument("--sparsegpt-b-damp", type=float, default=1e-6)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--berman-candidate-pool", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--fisher-general-lambda-u", type=float, default=1e-8)
    parser.add_argument("--fisher-general-lambda-d", type=float, default=1e-7)
    parser.add_argument("--fisher-general-mu", type=float, default=1e-4)
    parser.add_argument("--fisher-general-pool", type=int, default=8)
    parser.add_argument("--fisher-general-window", type=int, default=256)
    parser.add_argument("--fisher-general-group-pool", type=int, default=1)
    parser.add_argument("--fisher-general-max-group", type=int, default=16)
    parser.add_argument("--fisher-general-time-limit", type=float, default=60.)
    parser.add_argument("--fisher-general-gap", type=float, default=.01)
    parser.add_argument("--fisher-general-bound", type=float, default=10.)
    parser.add_argument("--fisher-general-log", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        RESULTS = args.output_dir
    args.sparsegpt_targets = tuple(float(x) for x in args.sparsegpt_targets.split(","))
    if (not args.sparsegpt_targets or any(not 0 < x < 1 for x in args.sparsegpt_targets)
            or tuple(sorted(set(args.sparsegpt_targets))) != args.sparsegpt_targets
            or args.sparsegpt_blocksize < 1 or min(args.sparsegpt_percdamp, args.sparsegpt_b_damp) < 0):
        parser.error("invalid Fisher-aware SparseGPT settings")
    args.device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    train, test_data = mnist()
    order = torch.randperm(len(test_data[0]), generator=torch.Generator().manual_seed(SPLIT_SEED))
    split = lambda index: (test_data[0][order[index]], test_data[1][order[index]])
    val_a, val_b, test = (loader(split(slice(0, 2500)), 512), loader(split(slice(2500, 5000)), 512),
                          loader(split(slice(5000, None)), 512))
    if args.method in ("berman", "all"):
        run_berman(args, train, val_b, test, args.fisher_n)
    if args.method == "fisher_sparsegpt":
        run_fisher_sparsegpt(args, train, val_b, test)
    for method in MOVES:
        if args.method in (method, "all"):
            run_greedy(args, method, train, val_a, val_b, test)
