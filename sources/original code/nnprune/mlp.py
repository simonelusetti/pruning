"""
Sigmoid MLP on two-moons (binary, Bernoulli likelihood) -- the pruningNN setting.

Boundary moves for a hidden sigmoid unit i of layer l
-----------------------------------------------------
* unit_sat  c=0 / c=1    b_i -> -inf / +inf.  For l >= 1 the inputs are sigmoid
                          outputs, strictly positive, so ANY incoming weight
                          w_ij -> -inf / +inf reaches the same endpoint: the
                          endpoint is unit-level even when the path is a single
                          weight.  Realised exactly by folding c * W_{l+1}[:, i]
                          into b_{l+1}; no large numbers are stored.
* unit_mean c=mean       whole fan-in -> 0 with b_i = logit(c): the unit is the
                          constant c, chosen as its training mean.  Same fold.
* unit_step (layer 0)     w_ij -> +-inf with sign-varying input x_j: the unit
                          becomes the Heaviside H(s x_j) and its other fan-in and
                          bias become unidentifiable (they are removed).
* unit_merge              a_i ~= alpha a_i2 + beta (OLS on training activations):
                          fold into unit i2's outgoing column and the next bias,
                          then remove i.  The duplicate-neuron singularity.
Weight moves ('zero') are generic (core.py).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .core import ACTIVE, REMOVED, STEP, Adapter, Move, State, zero_entries


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def make_moons_data(n=2500, noise=0.25, seed=123, val_frac=0.2, test_frac=0.2):
    """Train / validation / test.  The scaler is fit on TRAIN only."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_frac,
                                                random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=y_tmp)
    sc = StandardScaler().fit(X_tr)
    t = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    l = lambda a: torch.tensor(a, dtype=torch.float32)
    data = {"train": (t(X_tr), l(y_tr)), "val": (t(X_va), l(y_va)), "test": (t(X_te), l(y_te))}
    return data, sc


def moons_fresh_sample(scaler, n=8000, noise=0.25, seed=999):
    """Fresh draw from the same distribution (for expected-information diagnostics)."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return (torch.tensor(scaler.transform(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class MLPAdapter(Adapter):
    loss_type = "bernoulli"

    def __init__(self, data, widths=(2, 50, 50, 50, 1), step_layers=(0,)):
        super().__init__(data)
        self.widths = list(widths)
        self.L = len(widths) - 1
        self.prunable = [f"W{l}" for l in range(self.L)]
        self.site_layer = {f"W{l}": l for l in range(self.L)}
        # layers whose INPUTS change sign (standardised features): only there is
        # a single weight -> +-inf a Heaviside of one input rather than a constant
        self.step_layers = tuple(step_layers)

    # ---- construction -------------------------------------------------------
    def init_params(self, seed=123) -> dict:
        g = torch.Generator().manual_seed(seed)
        P = {}
        for l in range(self.L):
            fan_in, fan_out = self.widths[l], self.widths[l + 1]
            bound = 1.0 / np.sqrt(fan_in)
            P[f"W{l}"] = (torch.rand(fan_out, fan_in, generator=g) * 2 - 1) * bound
            P[f"b{l}"] = (torch.rand(fan_out, generator=g) * 2 - 1) * bound
        return P

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        struct = {}
        for l in range(self.L - 1):
            w = self.widths[l + 1]
            struct[f"type{l}"] = torch.full((w,), ACTIVE, dtype=torch.int8)
            struct[f"step_j{l}"] = torch.zeros(w, dtype=torch.long)
            struct[f"step_s{l}"] = torch.ones(w)
        return State(params, masks, struct)

    # ---- forward --------------------------------------------------------------
    def logits(self, P, S, X, return_acts=False):
        h = X
        acts = []
        out = None
        for l in range(self.L):
            z = h @ P[f"W{l}"].T + P[f"b{l}"]
            if l < self.L - 1:
                a = torch.sigmoid(z)
                t = S[f"type{l}"]
                if l in self.step_layers:
                    stepv = (h[..., S[f"step_j{l}"]] * S[f"step_s{l}"] > 0).to(a.dtype)
                    a = torch.where(t == STEP, stepv, a)
                a = a * (t != REMOVED).to(a.dtype)
                acts.append(a)
                h = a
            else:
                out = z.squeeze(-1)
        return (out, acts) if return_acts else out

    # ---- structural bookkeeping ----------------------------------------------
    def _kill(self, s: State, l: int, i: int) -> None:
        zero_entries(s, f"W{l}", (torch.arange(self.widths[l]) + i * self.widths[l]))
        zero_entries(s, f"b{l}", torch.tensor([i]))
        Wn = f"W{l + 1}"
        col = torch.zeros_like(s.masks[Wn])
        col[:, i] = True
        zero_entries(s, Wn, col)
        s.struct[f"type{l}"][i] = REMOVED

    def _fold_const(self, s: State, l: int, i: int, c: float) -> None:
        Wn, bn = f"W{l + 1}", f"b{l + 1}"
        s.params[bn] += s.params[Wn][:, i] * s.masks[Wn][:, i].to(s.params[Wn].dtype) * c
        self._kill(s, l, i)

    def canonicalize(self, s: State) -> None:
        changed = True
        while changed:
            changed = False
            for l in range(self.L - 1):
                t = s.struct[f"type{l}"]
                fan_out = s.masks[f"W{l + 1}"].any(0)
                fan_in = s.masks[f"W{l}"].any(1)
                alive = t != REMOVED
                for i in (alive & ~fan_out).nonzero().flatten().tolist():
                    self._kill(s, l, i)
                    changed = True
                t = s.struct[f"type{l}"]
                const = (t == ACTIVE) & ~fan_in & s.masks[f"W{l + 1}"].any(0)
                for i in const.nonzero().flatten().tolist():
                    b = s.params[f"b{l}"][i] if bool(s.masks[f"b{l}"][i]) else torch.tensor(0.0)
                    self._fold_const(s, l, i, float(torch.sigmoid(b)))
                    changed = True

    # ---- structural moves -------------------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        ctx = {"acts": acts, "act_mean": [a.mean(0) for a in acts], "merge_pairs": []}
        for l, a in enumerate(acts):
            t = s.struct[f"type{l}"]
            ids = (t == ACTIVE).nonzero().flatten()
            pairs = []
            if ids.numel() >= 2 and cfg.n_merge > 0:
                A = a[:, ids]
                Ac = A - A.mean(0)
                sd = Ac.pow(2).mean(0).sqrt().clamp_min(1e-12)
                C = (Ac.T @ Ac) / A.shape[0] / (sd[:, None] * sd[None, :])
                iu = torch.triu_indices(len(ids), len(ids), 1)
                vals = C[iu[0], iu[1]].abs()
                top = torch.argsort(vals, descending=True)[: cfg.n_merge]
                for t_ in top.tolist():
                    p, q = int(iu[0, t_]), int(iu[1, t_])
                    for i, i2 in ((p, q), (q, p)):
                        xi, xj = Ac[:, i], Ac[:, i2]
                        alpha = float((xi * xj).mean() / xj.pow(2).mean().clamp_min(1e-12))
                        beta = float(A[:, i].mean() - alpha * A[:, i2].mean())
                        pairs.append((int(ids[i]), int(ids[i2]), alpha, beta, float(vals[t_])))
            ctx["merge_pairs"].append(pairs)
        return ctx

    def structural_moves(self, s: State, ctx: dict, cfg) -> list:
        moves = []
        for l in range(self.L - 1):
            t = s.struct[f"type{l}"]
            mean = ctx["act_mean"][l]
            for i in (t != REMOVED).nonzero().flatten().tolist():
                for c in (0.0, 1.0):
                    moves.append(Move("unit_sat", f"h{l}", l, {"l": l, "i": i, "c": c}))
                moves.append(Move("unit_mean", f"h{l}", l, {"l": l, "i": i, "c": float(mean[i])}))
            if l in self.step_layers:
                for i in (t == ACTIVE).nonzero().flatten().tolist():
                    for j in range(self.widths[l]):
                        for sg in (1.0, -1.0):
                            moves.append(Move("unit_step", f"h{l}", l, {"l": l, "i": i, "j": j, "s": sg}))
            for (i, i2, alpha, beta, corr) in ctx["merge_pairs"][l]:
                moves.append(Move("unit_merge", f"h{l}", l,
                                  {"l": l, "i": i, "i2": i2, "alpha": alpha, "beta": beta, "corr": corr}))
        return moves

    def apply_structural(self, s: State, m: Move) -> bool:
        p = m.payload
        l, i = p["l"], p["i"]
        t = s.struct[f"type{l}"]
        if m.kind in ("unit_sat", "unit_mean"):
            if int(t[i]) == REMOVED:
                return False
            self._fold_const(s, l, i, p["c"])
            return True
        if m.kind == "unit_step":
            if int(t[i]) != ACTIVE or l not in self.step_layers:
                return False
            zero_entries(s, f"W{l}", torch.arange(self.widths[l]) + i * self.widths[l])
            zero_entries(s, f"b{l}", torch.tensor([i]))
            t[i] = STEP
            s.struct[f"step_j{l}"][i] = p["j"]
            s.struct[f"step_s{l}"][i] = p["s"]
            return True
        if m.kind == "unit_merge":
            i2 = p["i2"]
            if int(t[i]) != ACTIVE or int(t[i2]) != ACTIVE:
                return False
            Wn, bn = f"W{l + 1}", f"b{l + 1}"
            W, M = s.params[Wn], s.masks[Wn]
            col_i = W[:, i] * M[:, i].to(W.dtype)
            W[:, i2] = W[:, i2] * M[:, i2].to(W.dtype) + p["alpha"] * col_i
            M[:, i2] = M[:, i2] | M[:, i]
            s.params[bn] += p["beta"] * col_i
            self._kill(s, l, i)
            return True
        raise ValueError(m.kind)

    # ---- classical structured baseline ------------------------------------------
    @torch.no_grad()
    def classical_unit_order(self, s: State) -> list:
        """Mean-ablation order: Var(a_i) * ||W_{l+1}[:, i]||^2, smallest first."""
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        items = []
        for l, a in enumerate(acts):
            Wn = s.params[f"W{l + 1}"] * s.masks[f"W{l + 1}"]
            sc = a.var(0) * Wn.pow(2).sum(0)
            t = s.struct[f"type{l}"]
            items += [(float(sc[i]), (l, i)) for i in range(a.shape[1]) if int(t[i]) != REMOVED]
        return [u for _, u in sorted(items, key=lambda t: t[0])]

    @torch.no_grad()
    def mean_move(self, s: State, unit) -> Move:
        l, i = unit
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        return Move("unit_mean", f"h{l}", l, {"l": l, "i": i, "c": float(acts[l][:, i].mean())})

    # ---- Wanda input norms --------------------------------------------------------
    @torch.no_grad()
    def input_norms(self, s: State) -> dict:
        X, _ = self.data["train"]
        _, acts = self.logits(self.effective(s), s.struct, X, return_acts=True)
        ins = [X] + acts
        return {f"W{l}": ins[l].pow(2).sum(0).sqrt()[None, :] for l in range(self.L)}


# --------------------------------------------------------------------------
# Training (same recipe as pruningNN.ipynb: Adam 1e-3, full batch)
# --------------------------------------------------------------------------

def train_mlp(adapter: MLPAdapter, epochs=2500, lr=1e-3, seed=123, verbose=True):
    torch.manual_seed(seed)
    P = {k: v.requires_grad_(True) for k, v in adapter.init_params(seed).items()}
    s = adapter.make_state(P)
    X, y = adapter.data["train"]
    opt = torch.optim.Adam(list(P.values()), lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = adapter.loss_from_logits(adapter.logits(P, s.struct, X), y).mean()
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            with torch.no_grad():
                st = adapter.make_state(P)
                hist.append({"epoch": ep, "train_loss": loss.item(),
                             "val_loss": adapter.metrics(st, "val")["loss"]})
            if verbose and ep % 500 == 0:
                print(f"epoch {ep:5d}  train {hist[-1]['train_loss']:.4f}  val {hist[-1]['val_loss']:.4f}")
    return adapter.make_state(P), hist
