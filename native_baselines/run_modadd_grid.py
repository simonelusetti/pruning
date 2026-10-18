#!/usr/bin/env python3
"""The Berman SimpleNN/MNIST arms on the modular-addition transformer (NNpruning modadd).

Same protocol and Fisher objective as run_berman_mnist.py.  theta = every live parameter
(B_attn excluded: it is only a fold target of classic moves).
Weight moves: Berman's 10% step (smallest 1/2 theta_i^2 F_ii), zeroed or compensated within
its row.  A row = the parameters feeding one output coordinate: W_in, W_out, W_U with their
bias, W_O per residual coordinate (all heads), W_Q / W_K / W_V per (head, dim), W_E per token,
W_pos per position.  Rows of W_Q, W_K, W_E, W_pos enter nonlinearly: their Fisher is first order.
Neuron moves: attention heads, D = W_O[h], R = the other heads' W_O (exact Schur);
MLP units, D = W_out[:, i], R = other columns + b_out with F ~ E[a a^T] (x) E[M] (Kronecker),
under which the compensation is the regression of a_i on the surviving units and 1.
Every reported cost is 1/2 delta^T F delta of the applied delta under the exact Fisher.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))
sys.path.insert(0, str(ROOT / "native_baselines"))
from nnprune.core import ACTIVE, PruneConfig, refresh
from nnprune.experiments import TF_BASE, load_benchmark, split_validation
from run_berman_mnist import paired, save, schur, schur_invertible

RESULTS = ROOT / "native_baselines" / "results_modadd_grid"
NAMES = ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out",
         "W_U", "b_U")
COMMON = {"result_schema": 3, "model": "modadd_p59_one_layer_transformer",
          "dataset": "a + b mod 59", "checkpoint": "checkpoints/modadd_seed0.pt"}
MOVES = {"classic_weights": ["classic", "weight_zero"],
         "fisher": ["unit_fisher", "head_fisher", "weight_fisher", "weight_zero"],
         "fisher_weights": ["weight_fisher", "weight_zero"],
         "fisher_zero": ["weight_zero"],
         "fisher_zero_units": ["unit_zero", "head_zero", "weight_zero"]}


# --------------------------------------------------------------------------
# Flat parameter vector
# --------------------------------------------------------------------------

def flat(tensors):
    return torch.cat([tensors[n].flatten() for n in NAMES])


def unflat(vector, like):
    out, offset = {}, 0
    for n in NAMES:
        out[n] = vector[offset:offset + like[n].numel()].view_as(like[n])
        offset += like[n].numel()
    return out


def get_flat(adapter, state):
    return flat(adapter.effective(state)).double(), flat(state.masks)


def set_flat(state, theta, live):
    for n, p in unflat(theta, state.params).items():
        state.params[n] = p.to(state.params[n].dtype).clone()
    for n, m in unflat(live, state.masks).items():
        state.masks[n] = m.clone()


def row_view(P):
    """[rows, width] per group; one row = the parameters feeding one output coordinate."""
    per_head = lambda W: W.permute(0, 2, 1).reshape(-1, W.shape[1])
    return {"W_E": P["W_E"], "W_pos": P["W_pos"],
            "W_Q": per_head(P["W_Q"]), "W_K": per_head(P["W_K"]), "W_V": per_head(P["W_V"]),
            "W_O": P["W_O"].permute(2, 0, 1).reshape(P["W_O"].shape[2], -1),
            "W_in": torch.cat((P["W_in"], P["b_in"][:, None]), 1),
            "W_out": torch.cat((P["W_out"], P["b_out"][:, None]), 1),
            "W_U": torch.cat((P["W_U"].T, P["b_U"][:, None]), 1)}


def sqrt_fisher(p):
    """Rows s_c = sqrt(p_c)(e_c - p): sum_c s_c s_c^T = diag p - p p^T."""
    return p.sqrt()[:, :, None] * (torch.eye(p.shape[1], dtype=p.dtype) - p[:, None, :])


@torch.no_grad()
def fisher_blocks(adapter, P, struct, X):
    """Sums over (x, c) of g g^T for every row of row_view and for the whole W_O, where
    g_{x,c} = s_c^T d logits / d theta.  One layer, logits read at the last position: each
    gradient row is (class-dependent vector) x (input), so most class sums collapse first."""
    logits, cache = adapter.forward_full(P, struct, X)
    s = sqrt_fisher(logits.softmax(-1))                                   # [b, c, logit]
    alive = struct["head_alive"].to(P["W_O"].dtype)
    free = (~struct["head_fixed"]).to(P["W_O"].dtype)                      # fixed patterns: no score path
    x0 = cache["x0"]                                                       # [b, t, d]
    q = torch.einsum("btd,hde->bhte", x0, P["W_Q"])[:, :, -1]               # [b, h, e]
    k = torch.einsum("btd,hde->bhte", x0, P["W_K"])
    v = torch.einsum("btd,hde->bhte", x0, P["W_V"])
    a = cache["attn"][:, :, -1]                                            # [b, h, t]
    z, x1, act, x2 = cache["z"][:, :, -1], cache["x1"][:, -1], cache["act"][:, -1], cache["x2"][:, -1]
    u2 = s @ P["W_U"].T                                                    # d/dx2
    gp = (u2 @ P["W_out"]) * (act > 0)[:, None]                            # d/d MLP preactivation
    u1 = u2 + gp @ P["W_in"]                                               # d/dx1
    gz = torch.einsum("bcd,hed->bche", u1, P["W_O"]) * alive[:, None]      # d/dz
    ga = torch.einsum("bche,bhte->bcht", gz, v)                            # d/d attention
    gs = a[:, None] * (ga - (a[:, None] * ga).sum(-1, keepdim=True)) * free[:, None]  # d/d score
    r = 1 / math.sqrt(adapter.dh)
    gq = torch.einsum("bcht,bhte->bche", gs, k) * r
    gx0 = (torch.einsum("bht,bche,hde->bctd", a, gz, P["W_V"])
           + torch.einsum("bcht,bhe,hde->bctd", gs, q, P["W_K"]) * r)
    gx0[:, :, -1] += u1 + torch.einsum("bche,hde->bcd", gq, P["W_Q"])
    ones = torch.ones(len(X), 1, dtype=x0.dtype)
    outer = lambda w, inp: torch.einsum("br,bi,bj->rij", w, inp, inp)     # rows sharing one input
    per_head = lambda w, inp: torch.einsum("bhe,bhi,bhj->heij", w, inp, inp).flatten(0, 1)
    tokens = torch.nn.functional.one_hot(X, P["W_E"].shape[0]).to(x0.dtype)
    E = torch.einsum("btv,bctd->bcvd", tokens, gx0)
    w_K = torch.einsum("bcht,btd->bchd", gs, x0)
    za = (z * alive[:, None]).flatten(1)                                   # dead heads: no W_O gradient
    rows = {
        "W_E": torch.einsum("bcvi,bcvj->vij", E, E),
        "W_pos": torch.einsum("bcti,bctj->tij", gx0, gx0),
        "W_Q": per_head(gq.square().sum(1), x0[:, -1, None].expand(-1, adapter.H, -1)),
        "W_K": torch.einsum("bhe,bchi,bchj->heij", (q * r).square(), w_K, w_K).flatten(0, 1),
        "W_V": per_head(gz.square().sum(1), torch.einsum("bht,btd->bhd", a, x0)),
        "W_O": outer(u1.square().sum(1), za),
        "W_in": outer(gp.square().sum(1), torch.cat((x1, ones), 1)),
        "W_out": outer(u2.square().sum(1), torch.cat((act, ones), 1)),
        "W_U": outer(s.square().sum(1), torch.cat((x2, ones), 1))}
    M = torch.einsum("bco,bcp->bop", u1, u1)
    F_O = torch.einsum("bop,bi,bj->oipj", M, za, za).flatten(0, 1).flatten(1, 2)
    return rows, F_O


def fisher(adapter, state, X, blocks=True, chunk=128):
    """Exact model Fisher: the row blocks of every group, the full W_O block, and the diagonal
    (taken from the row blocks, which partition theta)."""
    P0 = {k: v.double() for k, v in adapter.effective(state).items()}
    theta = flat(P0)
    idx = row_view(unflat(torch.arange(theta.numel()), P0))
    rows, F_O = None, None
    for start in range(0, len(X), chunk):
        r, f = fisher_blocks(adapter, P0, state.struct, X[start:start + chunk])
        rows = r if rows is None else {g: rows[g] + r[g] for g in r}
        F_O = f if F_O is None else F_O + f
    n = len(X)
    rows = {g: F / n for g, F in rows.items()}
    diag = torch.zeros_like(theta)
    for g, I in idx.items():
        diag[I] = rows[g].diagonal(dim1=1, dim2=2)
    return diag, rows, F_O / n, idx


# --------------------------------------------------------------------------
# Moves.  Each proposal applies itself to a state; the loop canonicalises and counts.
# --------------------------------------------------------------------------

def berman_step(theta, live, diag, count):
    cost = (.5 * theta.square() * diag).masked_fill(~live, torch.inf)
    D = torch.zeros_like(live)
    D[torch.argsort(cost)[:min(count, int(live.sum()))]] = True
    return D


def weight_zero_candidates(theta, live, diag, k0, frac):
    D = berman_step(theta, live, diag, round(frac * k0))
    def apply(state):
        set_flat(state, theta.masked_fill(D, 0), live & ~D)
    return [{"kind": "weight_zero", "label": f"weight_zero:{int(D.sum())}", "apply": apply,
             "fisher_cost": .5 * float((theta.square() * diag)[D].sum())}]


def weight_fisher_candidates(theta, live, diag, rows, idx, k0, frac):
    D = berman_step(theta, live, diag, round(frac * k0))
    new, cost = theta.clone(), 0.
    for g, I in idx.items():
        delta, row_cost = schur(rows[g], theta[I], D[I])
        new[I] += delta
        cost += float(row_cost.sum())
    new[D] = 0
    def apply(state):
        set_flat(state, new, live & ~D)
    return [{"kind": "weight_fisher", "label": f"weight_fisher:{int(D.sum())}", "apply": apply,
             "fisher_cost": cost}]


def head_candidates(adapter, state, theta, live, F_O, idx, per_family, compensate):
    wo = idx["W_O"].flatten()
    th, lv = theta[wo], live[wo]
    head = torch.arange(len(wo)) % (adapter.H * adapter.dh) // adapter.dh
    heads = [h for h in range(adapter.H) if bool(state.struct["head_alive"][h])
             and bool(lv[head == h].any())]
    if not heads:
        return []
    D = torch.stack([(head == h) & lv for h in heads])
    if not compensate:
        delta = -th * D
        cost = .5 * ((delta @ F_O) * delta).sum(1)
    else:
        try:
            delta, cost = schur_invertible(F_O, th, D)
        except RuntimeError:
            delta, cost = schur(F_O.expand(len(heads), -1, -1), th.expand(len(heads), -1), D)
    saved = torch.tensor([sum(int(state.masks[n][h].sum()) for n in ("W_Q", "W_K", "W_V", "W_O"))
                          for h in heads], dtype=cost.dtype)
    kind = "head_fisher" if compensate else "head_zero"
    out = []
    for j in (cost / saved).argsort()[:per_family].tolist():
        new, new_live = theta.clone(), live.clone()
        new[wo] += delta[j]
        new[wo[D[j]]] = 0
        new_live[wo[D[j]]] = False
        out.append({"kind": kind, "label": f"{kind}:h{heads[j]}",
                    "apply": lambda state, new=new, new_live=new_live: set_flat(state, new, new_live),
                    "fisher_cost": float(cost[j])})
    return out


@torch.no_grad()
def unit_candidates(adapter, state, X, per_family, compensate):
    """MLP unit i: delta x2 (last position) = W_out[:, i] r(x), r = residual of a_i after the
    compensation (r = -a_i without it); exact cost 1/2 E[r^2 sum_c (s_c . W_U^T W_out[:, i])^2]."""
    P = {k: v.double() for k, v in adapter.effective(state).items()}
    logits, cache = adapter.forward_full(P, state.struct, X)
    units = ((state.struct["unit_type"] == ACTIVE) & state.masks["W_out"].any(0)).nonzero().flatten()
    if not units.numel():
        return []
    A = torch.cat((cache["act"][:, -1], torch.ones(len(X), 1, dtype=logits.dtype)), 1)
    q = (sqrt_fisher(logits.softmax(-1)) @ (P["W_U"].T @ P["W_out"][:, units])).square().sum(1)
    D = torch.zeros(len(units), A.shape[1], dtype=torch.bool)
    D[torch.arange(len(units)), units] = True
    if compensate:
        alive = torch.zeros(A.shape[1], dtype=A.dtype)
        alive[units] = 1
        alive[-1] = 1
        delta, _ = schur((A.T @ A / len(X)).expand(len(units), -1, -1),
                         alive.expand(len(units), -1), D)   # -1 at i, regression weights elsewhere
    else:
        delta = -D.to(A.dtype)
    cost = .5 * ((A @ delta.T).square() * q).mean(0)
    saved = (state.masks["W_out"][:, units].sum(0) + state.masks["W_in"][units].sum(1)
             + state.masks["b_in"][units]).to(cost.dtype)
    kind = "unit_fisher" if compensate else "unit_zero"
    out = []
    for j in (cost / saved).argsort()[:per_family].tolist():
        i, d = int(units[j]), delta[j]
        def apply(state, i=i, d=d):
            W = torch.cat((state.params["W_out"], state.params["b_out"][:, None]), 1).double()
            M = torch.cat((state.masks["W_out"], state.masks["b_out"][:, None]), 1)
            W = (W + torch.outer(W[:, i], d) * M) * M   # updates never revive pruned entries
            W[:, i] = 0
            state.params["W_out"] = W[:, :-1].to(state.params["W_out"].dtype).contiguous()
            state.params["b_out"] = W[:, -1].to(state.params["b_out"].dtype).contiguous()
            state.masks["W_out"][:, i] = False
        out.append({"kind": kind, "label": f"{kind}:n{i}", "apply": apply,
                    "fisher_cost": float(cost[j])})
    return out


def classic_candidates(adapter, state):
    cfg = PruneConfig(**dict(TF_BASE, weight_per_site=0), struct_kinds=("classic",),
                      verbose=False)
    shortlist = refresh(adapter, state, cfg)["struct_ranked"][:cfg.n_struct]
    return [{"kind": m.kind, "label": m.describe(),
             "apply": lambda state, m=m: adapter.apply_structural(state, m)}
            for m in adapter.prepare_structural_candidates(state, shortlist, cfg)]


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

def evaluate_row(adapter, state, reference, k0, started, step, **extra):
    val = adapter.per_sample(state, "valB")
    delta, se = paired(val, reference)
    test = adapter.metrics(state, "test")
    removed = k0 - adapter.k_eff(state)
    return {"step": step, "removed": removed, "frac_removed": removed / k0,
            "val_loss": float(val.mean()), "val_delta": delta, "val_se": se,
            "test_loss": test["loss"], "test_acc": test["acc"],
            "seconds": time.perf_counter() - started, **extra}


def protocol(args, **extra):
    return {"split_seed": 0, "val_a": 435, "val_b": 435, "test": 871, "rho": args.rho,
            "n_sigma": args.n_sigma, "fisher_subset_seed": 42, **extra}


def run_berman(args, adapter, state, X_fisher):
    k0 = adapter.k_eff(state)
    reference = adapter.per_sample(state, "valB")
    started = time.perf_counter()
    rows = [evaluate_row(adapter, state, reference, k0, started, 0)]
    base = rows[0]["val_loss"]
    failed = None
    n_targets = math.ceil(args.berman_max_frac / args.berman_step_frac)
    for step in range(1, n_targets + 1):
        target = min(step * args.berman_step_frac, args.berman_max_frac)
        theta, live = get_flat(adapter, state)
        diag = fisher(adapter, state, X_fisher, blocks=False)[0]
        count = max(0, round(target * k0) - (k0 - int(live.sum())))
        D = berman_step(theta, live, diag, count)
        set_flat(state, theta.masked_fill(D, 0), live & ~D)
        rows.append(evaluate_row(adapter, state, reference, k0, started, step))
        print(f"Berman n={len(X_fisher)} {target:.1%}: val {rows[-1]['val_loss']:.5f}", flush=True)
        if rows[-1]["val_delta"] > args.n_sigma * rows[-1]["val_se"] + args.rho * base:
            failed = len(rows) - 1
            break
    stop_index = failed - 1 if failed is not None else len(rows) - 1
    save(RESULTS / f"berman_iterative_n{len(X_fisher)}.json", {
        **COMMON, "status": "complete", "method": "fdist_iterative",
        "protocol": protocol(args, fisher_subset=len(X_fisher),
                             grid=args.berman_step_frac,
                             max_fraction=args.berman_max_frac),
        "stop_index": stop_index, "stopping_point": rows[stop_index],
        "rejected_candidate": rows[failed] if failed is not None else None,
        "history": rows})


def run_greedy(args, method, adapter, state, X_fisher):
    moves = MOVES[method]
    k0 = adapter.k_eff(state)
    ref_a, ref_b = adapter.per_sample(state, "valA"), adapter.per_sample(state, "valB")
    base = float(ref_b.mean())
    started = time.perf_counter()
    rows = [evaluate_row(adapter, state, ref_b, k0, started, 0)]
    stop_index, rejected = None, None
    record = lambda status: save(RESULTS / f"{method}.json", {
        **COMMON, "status": status, "method": method,
        "protocol": protocol(args, fisher_subset=len(X_fisher), weight_frac=args.weight_frac,
                             per_family=args.per_family, moves=moves),
        "stop_index": stop_index if stop_index is not None else len(rows) - 1,
        "stopping_point": rows[-1], "rejected_candidate": rejected, "history": rows,
        "move_counts": {k: sum(r.get("kind") == k for r in rows)
                        for k in sorted({r["kind"] for r in rows[1:]})}})
    blocks = "weight_fisher" in moves or "head_fisher" in moves or "head_zero" in moves
    for step in range(1, args.max_steps + 1):
        theta, live = get_flat(adapter, state)
        diag, F_rows, F_O, idx = fisher(adapter, state, X_fisher, blocks)
        proposals = weight_zero_candidates(theta, live, diag, k0, args.weight_frac)
        if "weight_fisher" in moves:
            proposals += weight_fisher_candidates(theta, live, diag, F_rows, idx, k0,
                                                  args.weight_frac)
        for kind, compensate in (("head_fisher", True), ("head_zero", False)):
            if kind in moves:
                proposals += head_candidates(adapter, state, theta, live, F_O, idx,
                                             args.per_family, compensate)
        for kind, compensate in (("unit_fisher", True), ("unit_zero", False)):
            if kind in moves:
                proposals += unit_candidates(adapter, state, X_fisher, args.per_family, compensate)
        if "classic" in moves:
            proposals += classic_candidates(adapter, state)
        k_now, evaluated = adapter.k_eff(state), []
        for proposal in proposals:
            candidate = state.clone()
            if proposal["apply"](candidate) is False:
                continue
            adapter.canonicalize(candidate)
            saved = k_now - adapter.k_eff(candidate)
            if saved > 0:
                evaluated.append((proposal, candidate, adapter.per_sample(candidate, "valA"), saved))
        if not evaluated:
            break
        best = min(evaluated, key=lambda item: float(item[2].mean()))[2]
        ties = [item for item in evaluated
                if paired(item[2], best)[0] <= args.n_sigma * paired(item[2], best)[1]]
        chosen, candidate, chosen_loss, saved = min(
            ties, key=lambda item: (-item[3], float(item[2].mean())))
        delta, se = paired(adapter.per_sample(candidate, "valB"), ref_b)
        extra = {"kind": chosen["kind"], "move": chosen["label"], "actual_savings": saved,
                 "val_a_loss": float(chosen_loss.mean()), "n_candidates": len(evaluated),
                 "n_ties": len(ties)}
        if "fisher_cost" in chosen:
            extra["fisher_cost"] = chosen["fisher_cost"]
        if delta > args.n_sigma * se + args.rho * base:
            stop_index = len(rows) - 1
            rejected = {**extra, "removed": k0 - adapter.k_eff(candidate),
                        "val_delta": delta, "val_se": se}
            break
        state = candidate
        rows.append(evaluate_row(adapter, state, ref_b, k0, started, step, **extra))
        record("running")
        print(f"{method} {step}: {chosen['label']} saved {saved}, removed "
              f"{rows[-1]['frac_removed']:.1%}, valB {rows[-1]['val_loss']:.5f}, "
              f"{rows[-1]['seconds']:.0f}s", flush=True)
    record("complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("berman", *MOVES, "all"), default="all")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--weight-frac", type=float, default=.1)
    parser.add_argument("--berman-step-frac", type=float, default=.01)
    parser.add_argument("--berman-max-frac", type=float, default=.2)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    if not 0 < args.berman_step_frac <= args.berman_max_frac <= 1:
        parser.error("require 0 < --berman-step-frac <= --berman-max-frac <= 1")
    torch.set_num_threads(args.threads)
    adapter, state = load_benchmark("modadd")
    adapter.data = split_validation(adapter.data)
    X_train = adapter.data["train"][0]
    order = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(42))
    if args.method in ("berman", "all"):
        run_berman(args, adapter, state.clone(), X_train)
        run_berman(args, adapter, state.clone(), X_train[order[:500]])
    for method in MOVES:
        if args.method in (method, "all"):
            run_greedy(args, method, adapter, state.clone(), X_train)


if __name__ == "__main__":
    main()
