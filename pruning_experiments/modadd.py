#!/usr/bin/env python3
"""Classic and Berman pruning on the modular-addition benchmark."""
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
from nnprune.core import PruneConfig, refresh
from nnprune.experiments import TF_BASE, load_benchmark, split_validation
from native_baselines.result_artifacts import write_json as save
from pruning_experiments.berman import paired

RESULTS = ROOT / "native_baselines" / "results_modadd_grid"
NAMES = ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out",
         "W_U", "b_U")
COMMON = {"result_schema": 3, "model": "modadd_p59_one_layer_transformer",
          "dataset": "a + b mod 59", "checkpoint": "checkpoints/modadd_seed0.pt"}
MOVES = {"classic": ("classic",),
         "classic_weights": ("classic", "weight_zero")}


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
    for step in range(1, args.max_steps + 1):
        proposals = []
        if "weight_zero" in moves:
            theta, live = get_flat(adapter, state)
            diag = fisher(adapter, state, X_fisher, blocks=False)[0]
            proposals += weight_zero_candidates(theta, live, diag, k0, args.weight_frac)
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


def main(argv=None):
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("berman", *MOVES, "all"), default="all")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rho", type=float, default=.05)
    parser.add_argument("--n-sigma", type=float, default=2.)
    parser.add_argument("--weight-frac", type=float, default=.1)
    parser.add_argument("--berman-step-frac", type=float, default=.01)
    parser.add_argument("--berman-max-frac", type=float, default=.2)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fisher-n", type=int)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        RESULTS = args.output_dir
    if not 0 < args.berman_step_frac <= args.berman_max_frac <= 1:
        parser.error("require 0 < --berman-step-frac <= --berman-max-frac <= 1")
    if args.method in ("berman", "all") and args.fisher_n not in (None, 500, 1740):
        parser.error("modular-addition Berman supports 500 or 1740 Fisher examples")
    torch.set_num_threads(args.threads)
    adapter, state = load_benchmark("modadd", args.device)
    adapter.data = split_validation(adapter.data)
    X_train = adapter.data["train"][0]
    order = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(42))
    if args.method in ("berman", "all"):
        if args.fisher_n is None or args.fisher_n == len(X_train):
            run_berman(args, adapter, state.clone(), X_train)
        if args.fisher_n is None or args.fisher_n == 500:
            run_berman(args, adapter, state.clone(), X_train[order[:500]])
    for method in MOVES:
        if args.method in (method, "all"):
            run_greedy(args, method, adapter, state.clone(), X_train)
