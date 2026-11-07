"""
Core of boundary pruning for neural networks: states, moves, the selection rule,
the greedy path, and the classical baselines.

Design in one paragraph
-----------------------
A *move* takes the network to a boundary of its parameter space: a weight to 0,
a unit to the constant it saturates to (sigmoid: b -> +-inf gives 1 / 0; for
positive inputs the same endpoint is reached by ANY incoming weight -> +-inf), a
unit to the constant it becomes when its whole fan-in goes to 0, a layer-0
sigmoid unit to the Heaviside limit H(s x_j) of one weight -> +-inf, a head to
uniform / fixed attention (the s -> 0 and "pattern constant in the input"
boundaries of softmax), and so on.  Limits are never stored as large numbers:
every limit is realised by its exact representative (a constant folded into the
downstream bias, a Heaviside unit, a fixed pattern).

Candidate moves are ranked by a local prediction (diagonal GGN/Fisher for
weights, a Fisher strategy, or training loss for classical structural moves).
The default decision validates candidates with a paired test:

  admissible   mean(l(f_m) - l(f_0)) <= n_sigma * SE + tol_abs + tol_frac mean(l(f_0))
  tie set      mean(l(f_m) - l(f_m*))  <= n_sigma * SE                   m* = best admissible
  tie-break    lexicographic on (k_eff removed, layer index, loss) -- flag `tie_break`

SE is always the SE of the per-sample *paired* difference.  The test split is
never used for any decision.

k_eff is the number of live parameters after structural cascades (a unit with no
live fan-out frees its fan-in; a unit with no live fan-in is a constant and is
folded downstream; ...).  It is the k of the BIC tie-break of emergentSR with the
induced nulls counted.  Continuous gauge nulls (ReLU scale, GL(d_h) of QK / OV)
are not subtracted from this count.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from .fisher import FisherStrategy, resolve_strategy

ACTIVE, REMOVED, STEP = 0, 1, 2


# --------------------------------------------------------------------------
# State and moves
# --------------------------------------------------------------------------

@dataclass
class State:
    params: dict                 # name -> tensor
    masks: dict                  # name -> bool tensor (same shape); False = pruned / structurally null
    struct: dict                 # adapter-specific structural status (tensors)

    def clone(self) -> "State":
        return State(
            {k: v.clone() for k, v in self.params.items()},
            {k: v.clone() for k, v in self.masks.items()},
            {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
             for k, v in self.struct.items()},
        )

    def to(self, device) -> "State":
        return State(
            {k: v.to(device) for k, v in self.params.items()},
            {k: v.to(device) for k, v in self.masks.items()},
            {k: (v.to(device) if torch.is_tensor(v) else copy.deepcopy(v))
             for k, v in self.struct.items()},
        )


@dataclass
class Move:
    kind: str                    # 'zero' or an adapter structural kind
    site: str                    # parameter name ('zero') or structural site label
    layer: int                   # depth index used by the layer tie-break
    payload: dict = field(default_factory=dict)
    pred_cost: float = float("nan")

    def describe(self) -> str:
        if self.kind == "fisher":
            repair = f"U={self.payload['n_updates']}"
            return (f"{self.kind}[{self.site}] D={self.payload['n_deleted']} "
                    f"{repair} K={self.payload['closure_savings']}")
        if self.kind == "zero":
            return f"zero[{self.site}] x{int(self.payload['idx'].numel())}"
        keys = {k: v for k, v in self.payload.items() if not torch.is_tensor(v)}
        return f"{self.kind}[{self.site}] {keys}"


def zero_entries(state: State, name: str, idx) -> None:
    """Mask entries (flat indices or a bool mask) of a parameter and zero them."""
    m = state.masks[name]
    p = state.params[name]
    if torch.is_tensor(idx) and idx.dtype == torch.bool:
        m[idx] = False
        p[idx] = 0
    else:
        m.view(-1)[idx] = False
        p.view(-1)[idx] = 0


# --------------------------------------------------------------------------
# Adapter base class
# --------------------------------------------------------------------------

class Adapter:
    """Model-specific pieces.  Subclasses implement `logits`, `canonicalize`,
    `structural_moves`, `apply_structural`, `refresh_extra`, `input_norms`."""

    loss_type = "categorical"          # or 'bernoulli'
    prunable: list = []                # weight tensors eligible for 'zero' moves
    site_layer: dict = {}              # prunable name -> depth index
    move_groups: dict[str, tuple[str, ...]] = {"classic": ()}
    move_kinds: tuple[str, ...] = ()

    def __init__(self, data: dict):
        self.data = data               # split -> (X, y)

    def to(self, device) -> "Adapter":
        self.data = {
            split: (value.to(device) if torch.is_tensor(value) else
                    tuple(x.to(device) if torch.is_tensor(x) else x for x in value))
            for split, value in self.data.items()
        }
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def resolve_struct_kinds(self, kinds) -> tuple[str, ...]:
        requested = ("classic",) if kinds is None else tuple(kinds)
        individual = ({kind for members in self.move_groups.values() for kind in members}
                      | set(self.move_kinds))
        unknown = [name for name in requested
                   if name not in self.move_groups and name not in individual]
        if unknown:
            raise ValueError(f"unknown structural move or group: {', '.join(unknown)}")
        resolved = []
        for name in requested:
            members = self.move_groups.get(name, (name,))
            resolved.extend(members)
        return tuple(dict.fromkeys(resolved))

    # ---- model ----------------------------------------------------------
    def logits(self, params: dict, struct: dict, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def effective(self, state: State) -> dict:
        return {k: state.params[k] * state.masks[k].to(state.params[k].dtype)
                for k in state.params}

    def loss_from_logits(self, z, y):
        if self.loss_type == "bernoulli":
            return F.binary_cross_entropy_with_logits(z, y.to(z.dtype), reduction="none")
        return F.cross_entropy(z, y, reduction="none")

    @torch.no_grad()
    def per_sample_loss(self, state: State, split: str, idx=None) -> torch.Tensor:
        X, y = self.data[split]
        if idx is not None:
            X, y = X[idx], y[idx]
        z = self.logits(self.effective(state), state.struct, X)
        return self.loss_from_logits(z, y)

    @torch.no_grad()
    def per_sample(self, state: State, split: str, criterion: str = "loss") -> torch.Tensor:
        """Per-sample loss, or per-sample 0/1 error, on a split."""
        X, y = self.data[split]
        z = self.logits(self.effective(state), state.struct, X)
        if criterion == "loss":
            return self.loss_from_logits(z, y)
        if self.loss_type == "bernoulli":
            return ((z > 0).long() != y.long()).double()
        return (z.argmax(-1) != y).double()

    @torch.no_grad()
    def metrics(self, state: State, split: str) -> dict:
        X, y = self.data[split]
        z = self.logits(self.effective(state), state.struct, X)
        loss = self.loss_from_logits(z, y).mean().item()
        if self.loss_type == "bernoulli":
            acc = ((z > 0).long() == y.long()).float().mean().item()
        else:
            acc = (z.argmax(-1) == y).float().mean().item()
        return {"loss": loss, "acc": acc}

    # ---- structure --------------------------------------------------------
    def extra_k(self, state: State) -> int:
        return 0

    def k_eff(self, state: State) -> int:
        return int(sum(int(m.sum()) for m in state.masks.values())) + self.extra_k(state)

    def canonicalize(self, state: State) -> None:
        pass

    def structural_moves(self, state: State, ctx: dict, cfg) -> list:
        return []

    def prepare_structural_candidates(self, state: State, moves: list, cfg) -> list:
        return moves

    def apply_structural(self, state: State, move: Move) -> bool:
        raise NotImplementedError(move.kind)

    def refresh_extra(self, state: State, cfg) -> dict:
        return {}

    def input_norms(self, state: State) -> dict:
        raise NotImplementedError

    # ---- moves --------------------------------------------------------------
    def apply(self, state: State, move: Move) -> Optional[State]:
        """Return a NEW canonicalised state, or None if the move is no longer valid."""
        s = state.clone()
        if move.kind == "zero":
            idx = move.payload["idx"]
            if not bool(s.masks[move.site].view(-1)[idx].any()):
                return None
            zero_entries(s, move.site, idx)
        else:
            if not self.apply_structural(s, move):
                return None
        if not all(bool(torch.isfinite(v).all()) for v in s.params.values()):
            return None
        self.canonicalize(s)
        return s


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TIE_BREAKS = ("k_eff", "layer", "k_eff_then_layer", "none")


@dataclass
class PruneConfig:
    # decision rule
    n_sigma: float = 2.0                 # the "2 sigma" of the paired tests
    tol_abs: float = 0.0                 # optional absolute slack on the anchored test (nats)
    tol_frac: float = 0.0                # optional fraction of the anchor's mean criterion
    tie_break: str = "k_eff_then_layer"  # one of TIE_BREAKS
    criterion: str = "loss"              # per-sample quantity of the paired tests: 'loss' | 'error'
    select_split: str = "val"            # split used to CHOOSE moves (admissibility, tie set): 'val' | 'valA' | 'train'
    stop_split: Optional[str] = None     # split of the anchored STOP test; None = same as select_split
    stop_mode: str = "halt"              # 'halt': end the path at the first stop-test failure;
                                         # 'mark': keep going, record the test, cut afterwards
                                         # (same stop point, but the whole curve is available)
    select_admissibility: bool = True    # False: the anchored test on select_split no longer filters
                                         # candidates; only the stop test on stop_split ends the path
    post_stop_steps: int = 0             # 'mark' mode: steps kept after the first stop-test failure
    max_frac: float = 1.0                # end the path once this fraction of k_eff is removed
    # candidate generation (geometry)
    weight_score: str = "obd"            # proposer for 'zero' moves: 'obd' | 'magnitude' | 'fdist' (Berman et al. 2026)
    weight_chunk: int = 1                # weights zeroed per 'zero' move
    weight_per_site: int = 8             # cheapest chunks proposed per prunable tensor
    use_struct: bool = True              # structural (boundary) moves on/off
    struct_kinds: Optional[tuple] = None  # move names/groups; None selects "classic"
    n_struct: int = 32                   # structural moves proposed per step (cheapest by screen)
    n_merge: int = 16                    # merge pairs screened per layer
    step_per_unit: Optional[int] = None  # strongest first-layer features; None = all
    refresh_every: int = 10              # steps between geometry refreshes
    screen_n: Optional[int] = None       # train points used by the structural screen (None = all)
    n_mc: int = 1                        # MC label samples for categorical Fisher
    fisher_batch: int = 256
    fisher_strategy: Optional[FisherStrategy | str] = None
    fisher_lambda_u: float = 1e-8
    fisher_lambda_d: float = 1e-7
    fisher_mu: float = 1e-4
    fisher_pool: int = 8
    fisher_time_limit: float = 60.0
    fisher_gap: float = 0.01
    fisher_bound: float = 10.0
    fisher_log: bool = False
    # path
    max_steps: int = 100000
    expand_on_fail: int = 4              # retry with this many times more candidates before stopping
    seed: int = 0
    verbose: bool = True
    log_every: int = 25

    def __post_init__(self):
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
        if self.criterion not in ("loss", "error"):
            raise ValueError("criterion must be 'loss' or 'error'")
        if self.weight_score not in ("obd", "magnitude", "fdist"):
            raise ValueError("weight_score must be 'obd', 'magnitude' or 'fdist'")
        if self.n_mc < 1:
            raise ValueError("n_mc must be positive")
        fisher_move = bool(self.struct_kinds and any(
            name in {"fisher", "fisher_general"} for name in self.struct_kinds))
        if fisher_move and any(name not in {"fisher", "fisher_general"}
                               for name in self.struct_kinds):
            raise ValueError("Fisher and Classic candidate pools cannot be mixed")
        if fisher_move and self.weight_per_site:
            raise ValueError("Fisher strategies already select scalar deletions")
        if self.fisher_strategy is not None and not fisher_move:
            raise ValueError("fisher_strategy requires a Fisher structural move")
        if self.fisher_strategy is None and fisher_move:
            self.fisher_strategy = resolve_strategy("fisher_general")
        if isinstance(self.fisher_strategy, str):
            self.fisher_strategy = resolve_strategy(self.fisher_strategy)
        if min(self.fisher_lambda_u, self.fisher_lambda_d, self.fisher_mu) < 0:
            raise ValueError("Fisher penalties must be non-negative")
        if self.fisher_pool < 1:
            raise ValueError("fisher_pool must be positive")
        if self.fisher_time_limit <= 0:
            raise ValueError("fisher_time_limit must be positive")
        if not 0 <= self.fisher_gap < 1 or self.fisher_bound <= 0:
            raise ValueError("invalid Fisher gap or compensation bound")
        if self.step_per_unit is not None and self.step_per_unit < 1:
            raise ValueError("step_per_unit must be positive")
        if self.stop_mode not in ("halt", "mark"):
            raise ValueError("stop_mode must be 'halt' or 'mark'")
        if self.n_sigma < 0 or self.tol_abs < 0 or self.tol_frac < 0:
            raise ValueError("n_sigma, tol_abs and tol_frac must be non-negative")
        if not 0 < self.max_frac <= 1:
            raise ValueError("max_frac must be in (0, 1]")
        if self.max_steps < 1 or self.refresh_every < 1:
            raise ValueError("max_steps and refresh_every must be positive")
        if min(self.weight_chunk, self.weight_per_site, self.n_struct,
               self.n_merge, self.post_stop_steps) < 0:
            raise ValueError("candidate budgets must be non-negative")
        if self.weight_per_site and self.weight_chunk < 1:
            raise ValueError("weight_chunk must be positive when weight moves are enabled")

    @property
    def stop(self) -> str:
        return self.stop_split or self.select_split


# --------------------------------------------------------------------------
# Geometry: diagonal GGN / Fisher and training gradient
# --------------------------------------------------------------------------

def diag_fisher_and_grad(adapter: Adapter, state: State, cfg: PruneConfig,
                         split: str = "train") -> tuple[dict, dict]:
    """Diagonal GGN (= Fisher for these likelihoods) of the SUMMED log-likelihood,
    and the gradient of the summed loss, for every prunable tensor.

    Bernoulli: exact, F_jj = sum_n p_n(1-p_n) (dz_n/dtheta_j)^2.
    Categorical: Monte Carlo with labels drawn from the model, n_mc draws.
    """
    X, y = adapter.data[split]
    eff = adapter.effective(state)
    names = list(adapter.prunable)
    fixed = {k: v for k, v in eff.items() if k not in names}
    W = {k: state.params[k].detach() for k in names}
    M = {k: state.masks[k].to(state.params[k].dtype) for k in names}
    struct = state.struct

    def out_single(w, x):
        P = dict(fixed)
        for k in names:
            P[k] = w[k] * M[k]
        return adapter.logits(P, struct, x.unsqueeze(0)).squeeze(0)

    fisher = {k: torch.zeros_like(W[k]) for k in names}
    grads = {k: torch.zeros_like(W[k]) for k in names}
    gen = torch.Generator(device=X.device).manual_seed(cfg.seed)
    batches = DataLoader(TensorDataset(X, y), batch_size=cfg.fisher_batch,
                         shuffle=False)
    for xb, yb in batches:
        with torch.no_grad():
            P = dict(fixed)
            for k in names:
                P[k] = W[k] * M[k]
            zb = adapter.logits(P, struct, xb)
        if adapter.loss_type == "bernoulli":
            jac = vmap(grad(lambda w, x: out_single(w, x)), in_dims=(None, 0))(W, xb)
            pb = torch.sigmoid(zb)
            wgt = pb * (1 - pb)
            for k in names:
                fisher[k] += torch.einsum("b,b...->...", wgt, jac[k] ** 2)
        else:
            probs = torch.softmax(zb, -1)
            for _ in range(cfg.n_mc):
                # torch.multinomial accepts only one- or two-dimensional input;
                # flatten any structured sample axes (for example GPT sequence
                # positions) while preserving the final class axis.
                yh = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=gen)
                yh = yh.squeeze(-1).reshape(probs.shape[:-1])
                g = vmap(grad(lambda w, x, t: torch.log_softmax(out_single(w, x), -1)
                              .gather(-1, t.unsqueeze(-1)).squeeze(-1).mean()),
                         in_dims=(None, 0, 0))(W, xb, yh)
                for k in names:
                    fisher[k] += (g[k] ** 2).sum(0) / cfg.n_mc

        Wg = {k: W[k].clone().requires_grad_(True) for k in names}
        P = dict(fixed)
        for k in names:
            P[k] = Wg[k] * M[k]
        loss = adapter.loss_from_logits(adapter.logits(P, struct, xb), yb).sum()
        batch_grads = torch.autograd.grad(loss, [Wg[k] for k in names])
        for k, value in zip(names, batch_grads):
            grads[k] += value.detach()
    return fisher, grads


def obd_scores(state: State, ctx: dict, names) -> dict:
    """Second-order predicted change of the summed training loss for theta_j -> 0:
    dL ~= -g_j theta_j + 0.5 F_jj theta_j^2  (OBD with the gradient term kept)."""
    out = {}
    for k in names:
        w = state.params[k]
        out[k] = -ctx["grad"][k] * w + 0.5 * ctx["fisher"][k] * w * w
    return out


# --------------------------------------------------------------------------
# Paired statistics and the selection rule
# --------------------------------------------------------------------------

def paired_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Mean and standard error of the per-sample paired difference a - b."""
    d = (a - b).double()
    n = d.numel()
    mean = float(d.mean())
    se = float(d.std(unbiased=True)) / math.sqrt(n) if n > 1 else float("inf")
    return mean, se


def _tie_key(tie_break: str):
    if tie_break == "k_eff":
        return lambda r: (-r["dk"], r["mean_crit"], r["mean_loss"])
    if tie_break == "layer":
        return lambda r: (r["layer"], r["mean_crit"], r["mean_loss"])
    if tie_break == "k_eff_then_layer":
        return lambda r: (-r["dk"], r["layer"], r["mean_crit"], r["mean_loss"])
    return lambda r: (r["mean_crit"], r["mean_loss"])


def select_move(results: list, cfg: PruneConfig):
    """Apply the anchored admissibility test, form the paired-n_sigma tie set
    around the best admissible candidate, and break ties lexicographically."""
    adm = [r for r in results if r["admissible"]]
    if not adm:
        return None, 0, 0
    best = min(adm, key=lambda r: (r["mean_crit"], r["mean_loss"]))
    if cfg.tie_break == "none":
        return best, len(adm), 1
    ties = []
    for r in adm:
        m, se = paired_stats(r["crit"], best["crit"])
        if m <= cfg.n_sigma * se or r is best:
            ties.append(r)
    chosen = min(ties, key=_tie_key(cfg.tie_break))
    return chosen, len(adm), len(ties)


# --------------------------------------------------------------------------
# Refresh, candidates, evaluation
# --------------------------------------------------------------------------

def refresh(adapter: Adapter, state: State, cfg: PruneConfig) -> dict:
    ctx = {}
    if cfg.weight_per_site > 0:
        if cfg.weight_score == "obd":
            ctx["fisher"], ctx["grad"] = diag_fisher_and_grad(adapter, state, cfg)
        elif cfg.weight_score == "fdist":
            from .fdist_baseline import _sqrtF_at, fisher_loader
            ctx["sqrtF"] = _sqrtF_at(adapter, state, fisher_loader(adapter))
    ctx.update(adapter.refresh_extra(state, cfg))
    ctx["struct_ranked"] = screen_structural(adapter, state, ctx, cfg) if cfg.use_struct else []
    return ctx


def _screen_idx(adapter: Adapter, cfg: PruneConfig):
    n = adapter.data["train"][0].shape[0]
    if cfg.screen_n is None or cfg.screen_n >= n:
        return None
    device = adapter.data["train"][0].device
    g = torch.Generator(device=device).manual_seed(cfg.seed + 7)
    return torch.randperm(n, generator=g, device=device)[: cfg.screen_n]


def screen_structural(adapter: Adapter, state: State, ctx: dict, cfg: PruneConfig) -> list:
    """Rank structural moves before the validation shortlist."""
    moves = adapter.structural_moves(state, ctx, cfg)
    allowed = adapter.resolve_struct_kinds(cfg.struct_kinds)
    moves = [m for m in moves if m.kind in allowed]
    idx = _screen_idx(adapter, cfg)
    base = adapter.per_sample_loss(state, "train", idx).mean().item()
    ranked = []
    for m in moves:
        if m.payload.get("screen_cost") == "fisher":
            if math.isfinite(m.pred_cost):
                ranked.append(m)
            continue
        s2 = adapter.apply(state, m)
        if s2 is None:
            continue
        m.pred_cost = adapter.per_sample_loss(s2, "train", idx).mean().item() - base
        if math.isfinite(m.pred_cost):
            ranked.append(m)
    ranked.sort(key=lambda m: m.pred_cost)
    return ranked


def weight_scores(state: State, ctx: dict, cfg: PruneConfig, names) -> dict:
    """Lower = proposed first."""
    if cfg.weight_score == "obd":
        return obd_scores(state, ctx, names)
    if cfg.weight_score == "magnitude":
        return {k: state.params[k].abs() for k in names}
    return {k: ctx["sqrtF"][k] * state.params[k].abs() for k in names}


def weight_candidates(adapter: Adapter, state: State, ctx: dict, cfg: PruneConfig,
                      expand: int = 1) -> list:
    if cfg.weight_per_site <= 0:
        return []
    scores = weight_scores(state, ctx, cfg, adapter.prunable)
    moves = []
    for name in adapter.prunable:
        live = state.masks[name].reshape(-1)
        if not bool(live.any()):
            continue
        sc = scores[name].reshape(-1)
        idx_live = live.nonzero().squeeze(1)
        order = idx_live[torch.argsort(sc[idx_live])]
        for c in range(cfg.weight_per_site * expand):
            ent = order[c * cfg.weight_chunk:(c + 1) * cfg.weight_chunk]
            if ent.numel() == 0:
                break
            moves.append(Move("zero", name, adapter.site_layer[name], {"idx": ent},
                              pred_cost=float(sc[ent].sum())))
    return moves


def candidates(adapter: Adapter, state: State, ctx: dict, cfg: PruneConfig,
               expand: int = 1) -> list:
    moves = weight_candidates(adapter, state, ctx, cfg, expand)
    if cfg.use_struct:
        shortlist = ctx["struct_ranked"][: cfg.n_struct * expand]
        moves += adapter.prepare_structural_candidates(state, shortlist, cfg)
    return moves


def evaluate_candidates(adapter: Adapter, state: State, moves: list, C0: torch.Tensor,
                        cfg: PruneConfig) -> list:
    """C0: per-sample criterion of the ORIGINAL model on cfg.select_split."""
    k_cur = adapter.k_eff(state)
    out = []
    tolerance = cfg.tol_abs + cfg.tol_frac * float(C0.mean())
    for m in moves:
        s2 = adapter.apply(state, m)
        if s2 is None:
            continue
        k_new = adapter.k_eff(s2)
        dk = k_cur - k_new
        if dk <= 0:
            continue
        losses = adapter.per_sample(s2, cfg.select_split, "loss")
        crit = losses if cfg.criterion == "loss" else adapter.per_sample(s2, cfg.select_split, "error")
        m0, se0 = paired_stats(crit, C0)
        out.append({
            "move": m, "state": s2, "k": k_new, "dk": dk, "layer": m.layer,
            "crit": crit, "mean_crit": float(crit.mean()), "mean_loss": float(losses.mean()),
            "anch_mean": m0, "anch_se": se0,
            "admissible": (m0 <= cfg.n_sigma * se0 + tolerance) or not cfg.select_admissibility,
        })
    return out


def evaluate_proxy(adapter: Adapter, state: State, moves: list, C0: torch.Tensor,
                   cfg: PruneConfig) -> list:
    k_cur = adapter.k_eff(state)
    for move in sorted(moves, key=lambda candidate: candidate.pred_cost):
        candidate = adapter.apply(state, move)
        if candidate is None:
            continue
        dk = k_cur - adapter.k_eff(candidate)
        if dk <= 0:
            continue
        losses = adapter.per_sample(candidate, cfg.select_split, "loss")
        criterion = (losses if cfg.criterion == "loss" else
                     adapter.per_sample(candidate, cfg.select_split, "error"))
        mean, se = paired_stats(criterion, C0)
        return [{"move": move, "state": candidate, "k": k_cur - dk, "dk": dk,
                 "layer": move.layer, "crit": criterion,
                 "mean_crit": float(criterion.mean()),
                 "mean_loss": float(losses.mean()), "anch_mean": mean,
                 "anch_se": se, "admissible": True}]
    return []


# --------------------------------------------------------------------------
# The greedy path
# --------------------------------------------------------------------------

def _row(adapter, state, k0, **extra) -> dict:
    tr = adapter.metrics(state, "train")
    va = adapter.metrics(state, "val")
    te = adapter.metrics(state, "test")
    k = adapter.k_eff(state)
    row = {"k_eff": k, "frac_removed": 1.0 - k / k0,
           "train_loss": tr["loss"], "train_acc": tr["acc"],
           "val_loss": va["loss"], "val_acc": va["acc"],
           "test_loss": te["loss"], "test_acc": te["acc"]}
    row.update(extra)
    return row


def prune_path(adapter: Adapter, state0: State, cfg: PruneConfig,
               label: str = "boundary") -> tuple[pd.DataFrame, State]:
    """Greedy boundary pruning with the paired-n_sigma rule.  Moves are chosen
    on cfg.select_split.  Stops when no proposed move is admissible there (after
    one expanded retry) or, if cfg.stop_split differs, when the chosen move fails
    the anchored test on cfg.stop_split (the move is then not applied)."""
    torch.manual_seed(cfg.seed)
    L0 = adapter.per_sample(state0, cfg.select_split, cfg.criterion)
    separate_stop = cfg.stop != cfg.select_split
    S0 = adapter.per_sample(state0, cfg.stop, cfg.criterion) if separate_stop else None
    stop_reference = S0 if separate_stop else L0
    stop_tolerance = cfg.tol_abs + cfg.tol_frac * float(stop_reference.mean())
    k0 = adapter.k_eff(state0)
    state = state0
    rows = [_row(adapter, state, k0, step=0, method=label, kind="", site="",
                 move="", layer=-1, dk=0, pred_cost=float("nan"), anch_mean=0.0, anch_se=0.0,
                 stop_mean=0.0, stop_se=0.0, stop_pass=True,
                 n_cands=0, n_adm=0, n_ties=0, seconds=0.0)]
    ctx, since = None, 0
    t0 = time.time()
    fails_at = None
    stop_state = None
    termination = "max_steps"
    trust_proxy = (cfg.fisher_strategy is not None and
                   cfg.fisher_strategy.decision == "proxy")
    for step in range(1, cfg.max_steps + 1):
        if fails_at is not None and step - fails_at > cfg.post_stop_steps:
            termination = "stop_rule"
            break
        if 1.0 - adapter.k_eff(state) / k0 >= cfg.max_frac:
            termination = "max_fraction"
            break
        if ctx is None or since >= cfg.refresh_every:
            ctx, since = refresh(adapter, state, cfg), 0
        moves = candidates(adapter, state, ctx, cfg)
        res = (evaluate_proxy(adapter, state, moves, L0, cfg) if trust_proxy else
               evaluate_candidates(adapter, state, moves, L0, cfg))
        chosen, n_adm, n_ties = ((res[0], len(res), len(res)) if trust_proxy and res else
                                 select_move(res, cfg))
        if chosen is None and cfg.expand_on_fail > 1 and (cfg.select_admissibility or not res):
            ctx, since = refresh(adapter, state, cfg), 0
            moves = candidates(adapter, state, ctx, cfg, expand=cfg.expand_on_fail)
            res = (evaluate_proxy(adapter, state, moves, L0, cfg) if trust_proxy else
                   evaluate_candidates(adapter, state, moves, L0, cfg))
            chosen, n_adm, n_ties = ((res[0], len(res), len(res))
                                     if trust_proxy and res else select_move(res, cfg))
        if chosen is None:
            termination = "no_admissible_candidate"
            if cfg.verbose:
                print(f"[{label}] stop at step {step}: no admissible move "
                      f"among {len(res)} candidates")
            break
        if separate_stop:
            sm, sse = paired_stats(adapter.per_sample(chosen["state"], cfg.stop, cfg.criterion), S0)
        else:
            sm, sse = chosen["anch_mean"], chosen["anch_se"]
        stop_pass = trust_proxy or bool(sm <= cfg.n_sigma * sse + stop_tolerance)
        if not stop_pass and fails_at is None:
            fails_at = step
            stop_state = state
        if not stop_pass and cfg.stop_mode == "halt":
            termination = "stop_rule"
            if cfg.verbose:
                print(f"[{label}] stop at step {step}: chosen move fails the "
                      f"anchored test on '{cfg.stop}'")
            break
        state = chosen["state"]
        since += 1
        mv = chosen["move"]
        if ctx is not None and ctx.get("fisher_moves"):
            ctx = None
        rows.append(_row(adapter, state, k0, step=step, method=label, kind=mv.kind,
                         site=mv.site, move=mv.describe(), layer=mv.layer,
                         dk=chosen["dk"], pred_cost=mv.pred_cost,
                         anch_mean=chosen["anch_mean"], anch_se=chosen["anch_se"],
                         stop_mean=sm, stop_se=sse, stop_pass=stop_pass,
                         n_cands=len(res), n_adm=n_adm, n_ties=n_ties,
                         n_deleted=mv.payload.get("n_deleted"),
                         n_updates=mv.payload.get("n_updates"),
                         closure_savings=mv.payload.get("closure_savings"),
                         search_objective=mv.payload.get("search_objective"),
                         solver_objective=mv.payload.get("solver_objective"),
                         solver_gap=mv.payload.get("solver_gap"),
                         solver_seconds=mv.payload.get("solver_seconds"),
                         compensation_bound_active=mv.payload.get(
                             "compensation_bound_active"),
                         seconds=time.time() - t0))
        if cfg.verbose and step % cfg.log_every == 0:
            r = rows[-1]
            print(f"[{label}] step {step:5d}  k_eff {r['k_eff']:6d} "
                  f"({100 * r['frac_removed']:5.1f}% removed)  val {r['val_loss']:.4f}  "
                  f"test {r['test_loss']:.4f}  last={mv.describe()}  "
                  f"ties {n_ties}/{n_adm}  {r['seconds']:.0f}s")
    df = pd.DataFrame(rows)
    # path semantics: the method stops before the first stop-test failure
    bad = ~df["stop_pass"].values.astype(bool)
    first_fail = int(np.argmax(bad)) if bad.any() else len(df)
    df["within_stop"] = np.arange(len(df)) < first_fail
    df.attrs.update({
        "termination_reason": termination,
        "select_split": cfg.select_split,
        "stop_split": cfg.stop,
        "criterion": cfg.criterion,
        "n_sigma": cfg.n_sigma,
        "tol_abs": cfg.tol_abs,
        "tol_frac": cfg.tol_frac,
        "tie_break": cfg.tie_break,
        "stop_mode": cfg.stop_mode,
        "select_admissibility": cfg.select_admissibility,
        "decision": "proxy" if trust_proxy else "validate",
        "max_frac": cfg.max_frac,
        "select_reference": float(L0.mean()),
        "stop_reference": float(stop_reference.mean()),
    })
    return df, stop_state if stop_state is not None else state


# --------------------------------------------------------------------------
# Classical baselines (one-shot rankings, evaluated along a sparsity grid)
# --------------------------------------------------------------------------

def baseline_scores(adapter: Adapter, state0: State, method: str, cfg: PruneConfig,
                    ctx: Optional[dict] = None) -> dict:
    """Lower score = pruned first.
    magnitude |w|;  obd -g w + F w^2 / 2;  wanda |w| * ||input_j||  (global ranking,
    not Wanda's per-row grouping);  random."""
    names = adapter.prunable
    if method == "magnitude":
        return {k: state0.params[k].abs() for k in names}
    if method == "obd":
        if ctx is None:
            f, g = diag_fisher_and_grad(adapter, state0, cfg)
            ctx = {"fisher": f, "grad": g}
        return obd_scores(state0, ctx, names)
    if method == "wanda":
        norms = adapter.input_norms(state0)
        return {k: state0.params[k].abs() * norms[k] for k in names}
    if method == "random":
        device = next(iter(state0.params.values())).device
        g = torch.Generator(device=device).manual_seed(cfg.seed + 11)
        return {k: torch.rand(state0.params[k].shape, generator=g, device=device) for k in names}
    raise ValueError(method)


def _anchored_columns(adapter: Adapter, s: State, ref: dict, cfg: PruneConfig) -> dict:
    """Anchored paired test against the original model, for BOTH criteria."""
    out = {}
    for crit in ("loss", "error"):
        m0, se0 = paired_stats(adapter.per_sample(s, cfg.stop, crit), ref[crit])
        tolerance = cfg.tol_abs + cfg.tol_frac * float(ref[crit].mean())
        out[f"anch_mean_{crit}"] = m0
        out[f"anch_se_{crit}"] = se0
        out[f"admissible_{crit}"] = bool(m0 <= cfg.n_sigma * se0 + tolerance)
    return out


def _within_tolerance(df: pd.DataFrame) -> pd.DataFrame:
    """Path semantics: the admissible region ends at the first failure."""
    for crit in ("loss", "error"):
        bad = ~df[f"admissible_{crit}"].values
        first_fail = int(np.argmax(bad)) if bad.any() else len(df)
        df[f"within_tolerance_{crit}"] = np.arange(len(df)) < first_fail
    return df


def baseline_path(adapter: Adapter, state0: State, method: str, cfg: PruneConfig,
                  fracs=None, ctx: Optional[dict] = None) -> pd.DataFrame:
    """Prune the globally lowest-scored fraction of prunable weights (no
    compensation, no retraining), canonicalise, evaluate.  `frac_removed` is in
    k_eff units, so cascades are credited exactly as for the boundary method."""
    if fracs is None:
        fracs = np.round(np.concatenate([np.arange(0, 0.5, 0.02),
                                         np.arange(0.5, 0.99, 0.01)]), 4)
    scores = baseline_scores(adapter, state0, method, cfg, ctx)
    names = adapter.prunable
    flat_s, flat_name, flat_idx = [], [], []
    for j, k in enumerate(names):
        live = state0.masks[k].reshape(-1)
        idx = live.nonzero().squeeze(1)
        flat_s.append(scores[k].reshape(-1)[idx])
        flat_name.append(torch.full_like(idx, j))
        flat_idx.append(idx)
    flat_s = torch.cat(flat_s)
    flat_name = torch.cat(flat_name)
    flat_idx = torch.cat(flat_idx)
    order = torch.argsort(flat_s)
    total = order.numel()
    ref = {c: adapter.per_sample(state0, cfg.stop, c) for c in ("loss", "error")}
    k0 = adapter.k_eff(state0)
    rows = []
    for f in fracs:
        n = int(round(f * total))
        s = state0.clone()
        sel = order[:n]
        for j, k in enumerate(names):
            idx = flat_idx[sel[flat_name[sel] == j]]
            if idx.numel():
                zero_entries(s, k, idx)
        adapter.canonicalize(s)
        rows.append(_row(adapter, s, k0, method=method, weight_frac=float(f), n_zeroed=n,
                         **_anchored_columns(adapter, s, ref, cfg)))
    return _within_tolerance(pd.DataFrame(rows))


def unit_baseline_path(adapter: Adapter, state0: State, cfg: PruneConfig,
                       label: str = "mean_ablation") -> pd.DataFrame:
    """Classical structured baseline: remove units (and heads) one at a time in a
    one-shot order (smallest variance of their contribution first), each replaced
    by its current mean and folded downstream (mean-replacement pruning).  No
    selection rule, no other move types."""
    order = adapter.classical_unit_order(state0)
    ref = {c: adapter.per_sample(state0, cfg.stop, c) for c in ("loss", "error")}
    k0 = adapter.k_eff(state0)
    rows = [_row(adapter, state0, k0, method=label, n_units=0,
                 **_anchored_columns(adapter, state0, ref, cfg))]
    s = state0
    for j, u in enumerate(order, 1):
        s2 = adapter.apply(s, adapter.mean_move(s, u))
        if s2 is None:
            continue
        s = s2
        rows.append(_row(adapter, s, k0, method=label, n_units=j,
                         **_anchored_columns(adapter, s, ref, cfg)))
    return _within_tolerance(pd.DataFrame(rows))


def summarize(paths: dict, baselines: dict, criterion: str = "loss") -> pd.DataFrame:
    """One row per method at its own stopping point (boundary paths: last step;
    baselines: last grid point before the first anchored-test failure under
    `criterion`)."""
    rows = []
    for name, df in paths.items():
        r = df[df["within_stop"]].iloc[-1] if "within_stop" in df else df.iloc[-1]
        rows.append({"method": name, "frac_removed": r["frac_removed"], "k_eff": int(r["k_eff"]),
                     "val_loss": r["val_loss"], "test_loss": r["test_loss"],
                     "test_acc": r["test_acc"], "steps": int(r["step"])})
    for name, df in baselines.items():
        ok = df[df[f"within_tolerance_{criterion}"]]
        r = ok.iloc[-1] if len(ok) else df.iloc[0]
        rows.append({"method": name, "frac_removed": r["frac_removed"], "k_eff": int(r["k_eff"]),
                     "val_loss": r["val_loss"], "test_loss": r["test_loss"],
                     "test_acc": r["test_acc"], "steps": np.nan})
    return pd.DataFrame(rows).sort_values("frac_removed", ascending=False).reset_index(drop=True)
