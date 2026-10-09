"""
CNN adapter for Fashion-MNIST (28x28), same moves as cnn.py, one change in the model:
a 2x2 max-pool after conv1.

    x (1x28x28) -> conv1 1->16, 3x3, pad 1 -> ReLU -> max-pool 2 -> a1 (16 x 14 x 14)
                -> conv2 16->32, 3x3, pad 1, + b2 + border-aware B2 -> ReLU -> GAP -> fc 32->10

"Slices" of conv1 are the pooled maps a1, i.e. what conv2 actually reads, so every
move is defined (and exact) on them; the pool needs no special treatment.
Statistics for the structural moves (means, correlations, the spatial-merge
regressions) are computed on `stats_n` training images.

Data: the official Fashion-MNIST files (github.com/zalandoresearch/fashion-mnist),
read from ../data/fashion.  60 000 training images -> 50 000 to train the network,
10 000 held out; the held-out part gives valA (2 000, used to choose moves) and
valB (8 000, used only by the stop test).  The 10 000 official test images are
the test split.  The split called "train" inside the adapter is a fixed 5 000-image
subset of the 50 000: her core uses it for the Fisher diagonal, the structural
screen and the train-selected arm.
"""
from __future__ import annotations

import gzip
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "NNpruning"))
from nnprune.core import ACTIVE, REMOVED, Adapter, Move, State, zero_entries  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "fashion")
HM = 14                                                    # map size seen by conv2
_REG = torch.tensor([0] + [1] * (HM - 2) + [2])
REP = [0, HM // 2, HM - 1]

NAIVE_KINDS = ("chan_mean_in", "chan_merge_pw", "gap_mean", "gap_merge")
CNN_KINDS = NAIVE_KINDS + ("chan_mean_s", "chan_mean_r", "chan_merge_sp")


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    nd = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)


def load_fashion(seed=0, n_valA=2000, n_prune_train=5000):
    Xtr = _idx(os.path.join(DATA, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    ytr = _idx(os.path.join(DATA, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    Xte = _idx(os.path.join(DATA, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    yte = _idx(os.path.join(DATA, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    Xtr = torch.tensor(Xtr)[:, None]
    Xte = torch.tensor(Xte)[:, None]
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ytr), generator=g)
    held, fit = perm[:10000], perm[10000:]
    sub = fit[torch.randperm(len(fit), generator=g)[:n_prune_train]]
    data = {"train": (Xtr[sub], ytr[sub]),
            "val": (Xtr[held], ytr[held]),
            "valA": (Xtr[held[:n_valA]], ytr[held[:n_valA]]),
            "valB": (Xtr[held[n_valA:]], ytr[held[n_valA:]]),
            "test": (Xte, yte)}
    fit_data = (Xtr[fit], ytr[fit])
    return data, fit_data


def region_pattern(K: torch.Tensor) -> torch.Tensor:
    """Response of kernels K (..., 3, 3) to an HMxHM map of ones with zero padding,
    at the 9 border regions: (..., 3, 3)."""
    lead = K.shape[:-2]
    resp = F.conv2d(torch.ones(1, 1, HM, HM), K.reshape(-1, 1, 3, 3), padding=1)[0]
    return resp[:, REP][:, :, REP].reshape(*lead, 3, 3)


class CNNAdapterFM(Adapter):
    loss_type = "categorical"

    def __init__(self, data, C1=16, C2=32, n_classes=10, stats_n=2000):
        super().__init__(data)
        self.C1, self.C2, self.K, self.stats_n = C1, C2, n_classes, stats_n
        self.prunable = ["W1", "W2", "Wf"]
        self.site_layer = {"W1": 0, "W2": 1, "Wf": 2}

    def init_params(self, seed=0) -> dict:
        g = torch.Generator().manual_seed(seed)
        C1, C2 = self.C1, self.C2
        return {"W1": torch.randn(C1, 1, 3, 3, generator=g) * (2 / 9) ** 0.5, "b1": torch.zeros(C1),
                "W2": torch.randn(C2, C1, 3, 3, generator=g) * (2 / (C1 * 9)) ** 0.5,
                "b2": torch.zeros(C2), "B2": torch.zeros(C2, 3, 3),
                "Wf": torch.randn(self.K, C2, generator=g) * (1 / C2) ** 0.5, "bf": torch.zeros(self.K)}

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        masks["B2"][:] = False
        params["B2"].zero_()
        struct = {"type0": torch.full((self.C1,), ACTIVE, dtype=torch.int8),
                  "type1": torch.full((self.C2,), ACTIVE, dtype=torch.int8)}
        return State(params, masks, struct)

    # ---- forward -------------------------------------------------------------------
    def conv1_out(self, P, S, X):
        a1 = F.max_pool2d(F.relu(F.conv2d(X, P["W1"], P["b1"], padding=1)), 2)
        return a1 * (S["type0"] != REMOVED).to(a1.dtype)[None, :, None, None]

    def logits(self, P, S, X, return_acts=False):
        a1 = self.conv1_out(P, S, X)
        z2 = F.conv2d(a1, P["W2"], P["b2"], padding=1) + P["B2"][:, _REG][:, :, _REG]
        a2 = F.relu(z2) * (S["type1"] != REMOVED).to(z2.dtype)[None, :, None, None]
        g = a2.mean((2, 3))
        out = g @ P["Wf"].T + P["bf"]
        return (out, a1, g) if return_acts else out

    # ---- bookkeeping (as in cnn.py) ---------------------------------------------------
    def _activate_B2(self, s):
        s.masks["B2"][s.struct["type1"] != REMOVED] = True

    def _kill0(self, s, j):
        zero_entries(s, "W1", torch.arange(9) + j * 9)
        zero_entries(s, "b1", torch.tensor([j]))
        col = torch.zeros_like(s.masks["W2"]); col[:, j] = True
        zero_entries(s, "W2", col)
        s.struct["type0"][j] = REMOVED

    def _kill1(self, s, o):
        row = torch.zeros_like(s.masks["W2"]); row[o] = True
        zero_entries(s, "W2", row)
        zero_entries(s, "b2", torch.tensor([o]))
        rowB = torch.zeros_like(s.masks["B2"]); rowB[o] = True
        zero_entries(s, "B2", rowB)
        colf = torch.zeros_like(s.masks["Wf"]); colf[:, o] = True
        zero_entries(s, "Wf", colf)
        s.struct["type1"][o] = REMOVED

    def _eff(self, s, name):
        return s.params[name] * s.masks[name].to(s.params[name].dtype)

    def canonicalize(self, s: State) -> None:
        changed = True
        while changed:
            changed = False
            t0, t1 = s.struct["type0"], s.struct["type1"]
            for j in range(self.C1):
                if int(t0[j]) == REMOVED:
                    continue
                if not bool(s.masks["W2"][:, j].any()):
                    self._kill0(s, j); changed = True; continue
                if not bool(s.masks["W1"][j].any()):
                    c = max(float(self._eff(s, "b1")[j]), 0.0)    # relu(b1), unchanged by the max-pool
                    if c != 0.0:
                        self._activate_B2(s)
                        s.params["B2"] += c * region_pattern(self._eff(s, "W2")[:, j])
                    self._kill0(s, j); changed = True
            for o in range(self.C2):
                if int(t1[o]) == REMOVED:
                    continue
                if not bool(s.masks["Wf"][:, o].any()):
                    self._kill1(s, o); changed = True; continue
                if not bool(s.masks["W2"][o].any()):
                    z = self._eff(s, "b2")[o] + self._eff(s, "B2")[o][_REG][:, _REG]
                    s.params["bf"] += float(F.relu(z).mean()) * self._eff(s, "Wf")[:, o]
                    self._kill1(s, o); changed = True

    # ---- statistics for the structural moves ----------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X = self.data["train"][0][: self.stats_n]
        P = self.effective(s)
        _, a1, g = self.logits(P, s.struct, X, return_acts=True)
        n = a1.shape[0]
        alive0 = (s.struct["type0"] == ACTIVE).nonzero().flatten().tolist()
        alive1 = (s.struct["type1"] == ACTIVE).nonzero().flatten().tolist()
        W2 = P["W2"]
        ctx = {"alive0": alive0, "alive1": alive1}
        kinds = set(cfg.struct_kinds) if cfg.struct_kinds is not None else set(CNN_KINDS)
        if not cfg.use_struct:
            kinds = set()
        need_contrib = bool(kinds & {"chan_mean_s", "chan_mean_r", "chan_merge_sp"})
        want_sp = "chan_merge_sp" in kinds
        ctx["c_in"] = {j: float(a1[:, j].mean()) for j in alive0}
        mean_s, mean_r, sp = {}, {}, []
        if want_sp:
            patches = F.unfold(a1, 3, padding=1).view(n, self.C1, 9, HM * HM)
            patches = patches.permute(0, 3, 1, 2).reshape(n * HM * HM, self.C1, 9)
            Z = torch.cat([patches[:, alive0], torch.ones(n * HM * HM, len(alive0), 1)], 2)  # (N, m, 10)
            Zf = Z.reshape(Z.shape[0], -1)                                                     # (N, 10m)
            G = torch.einsum("nmi,nmj->mij", Z, Z).double() + 1e-6 * torch.eye(10, dtype=torch.float64)
        for j in (alive0 if need_contrib else []):
            Cj = F.conv2d(a1[:, j:j + 1], W2[:, j:j + 1], None, padding=1)          # (n, C2, HM, HM)
            mean_s[j] = Cj.mean((0, 2, 3))
            m = Cj.mean(0)
            R = torch.zeros(self.C2, 3, 3)
            for r in range(3):
                for c in range(3):
                    R[:, r, c] = m[:, _REG == r][:, :, _REG == c].mean((1, 2))
            mean_r[j] = R
            if want_sp:
                T = Cj.permute(0, 2, 3, 1).reshape(-1, self.C2)                     # (N, C2)
                tt = float((T.double() - T.double().mean(0)).pow(2).sum())
                B = (Zf.T @ T).double().reshape(len(alive0), 10, self.C2)           # Z_i^T T for every i
                t2 = float(T.double().pow(2).sum())
                best = None
                for k, i in enumerate(alive0):
                    if i == j:
                        continue
                    sol = torch.linalg.solve(G[k], B[k])                              # (10, C2)
                    rss = t2 - float((B[k] * sol).sum())
                    r2 = 1 - rss / max(tt, 1e-12)
                    if best is None or r2 > best[0]:
                        best = (r2, i, sol)
                if best is not None:
                    r2, i, sol = best
                    sp.append((j, i, sol[:9].T.float().reshape(self.C2, 3, 3), sol[9].float(), r2))
        ctx["mean_s"], ctx["mean_r"], ctx["sp"] = mean_s, mean_r, sp
        # pointwise pairs
        pairs = []
        if len(alive0) >= 2 and cfg.n_merge > 0:
            A = a1.permute(1, 0, 2, 3).reshape(self.C1, -1)[alive0]
            Ac = A - A.mean(1, keepdim=True)
            sd = Ac.pow(2).mean(1).sqrt().clamp_min(1e-12)
            Cm = (Ac @ Ac.T) / Ac.shape[1] / (sd[:, None] * sd[None, :])
            iu = torch.triu_indices(len(alive0), len(alive0), 1)
            vals = Cm[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                p_, q_ = int(iu[0, t_]), int(iu[1, t_])
                for i, i2 in ((p_, q_), (q_, p_)):
                    alpha = float((Ac[i] * Ac[i2]).mean() / Ac[i2].pow(2).mean().clamp_min(1e-12))
                    beta = float(A[i].mean() - alpha * A[i2].mean())
                    pairs.append((alive0[i], alive0[i2], alpha, beta))
        ctx["pw_pairs"] = pairs
        ctx["g_mean"] = g.mean(0)
        gp = []
        if len(alive1) >= 2 and cfg.n_merge > 0:
            Gm = g[:, alive1].T
            Gc = Gm - Gm.mean(1, keepdim=True)
            sd = Gc.pow(2).mean(1).sqrt().clamp_min(1e-12)
            Cg = (Gc @ Gc.T) / Gc.shape[1] / (sd[:, None] * sd[None, :])
            iu = torch.triu_indices(len(alive1), len(alive1), 1)
            vals = Cg[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                p_, q_ = int(iu[0, t_]), int(iu[1, t_])
                for o, o2 in ((p_, q_), (q_, p_)):
                    alpha = float((Gc[o] * Gc[o2]).mean() / Gc[o2].pow(2).mean().clamp_min(1e-12))
                    beta = float(Gm[o].mean() - alpha * Gm[o2].mean())
                    gp.append((alive1[o], alive1[o2], alpha, beta))
        ctx["gap_pairs"] = gp
        return ctx

    def structural_moves(self, s, ctx, cfg) -> list:
        mv = []
        for j in ctx["alive0"]:
            mv.append(Move("chan_mean_in", "c1", 0, {"j": j, "c": ctx["c_in"][j]}))
            if j in ctx["mean_s"]:
                mv.append(Move("chan_mean_s", "c1", 0, {"j": j, "db2": ctx["mean_s"][j]}))
                mv.append(Move("chan_mean_r", "c1", 0, {"j": j, "dB2": ctx["mean_r"][j]}))
        for (j, i, alpha, beta) in ctx["pw_pairs"]:
            mv.append(Move("chan_merge_pw", "c1", 0, {"j": j, "i": i, "alpha": alpha, "beta": beta}))
        for (j, i, V, b, r2) in ctx["sp"]:
            mv.append(Move("chan_merge_sp", "c1", 0, {"j": j, "i": i, "V": V, "b": b, "r2": r2}))
        for o in ctx["alive1"]:
            mv.append(Move("gap_mean", "c2", 1, {"o": o, "g": float(ctx["g_mean"][o])}))
        for (o, o2, alpha, beta) in ctx["gap_pairs"]:
            mv.append(Move("gap_merge", "c2", 1, {"o": o, "o2": o2, "alpha": alpha, "beta": beta}))
        return mv

    def apply_structural(self, s, m) -> bool:
        p = m.payload
        t0, t1 = s.struct["type0"], s.struct["type1"]
        if m.site == "c1":
            j = p["j"]
            if int(t0[j]) != ACTIVE:
                return False
            W2e = self._eff(s, "W2")
            if m.kind == "chan_mean_in":
                s.params["b2"] += p["c"] * W2e[:, j].sum((1, 2))
            elif m.kind == "chan_mean_s":
                s.params["b2"] += p["db2"]
            elif m.kind == "chan_mean_r":
                self._activate_B2(s)
                s.params["B2"] += p["dB2"]
            elif m.kind in ("chan_merge_pw", "chan_merge_sp"):
                i = p["i"]
                if int(t0[i]) != ACTIVE:
                    return False
                if m.kind == "chan_merge_pw":
                    s.params["W2"][:, i] = W2e[:, i] + p["alpha"] * W2e[:, j]
                    s.masks["W2"][:, i] |= s.masks["W2"][:, j]
                    s.params["b2"] += p["beta"] * W2e[:, j].sum((1, 2))
                else:
                    s.params["W2"][:, i] = W2e[:, i] + p["V"]
                    s.masks["W2"][t1 != REMOVED, i] = True
                    s.params["b2"] += p["b"]
            else:
                raise ValueError(m.kind)
            self._kill0(s, j)
            return True
        if m.site == "c2":
            o = p["o"]
            if int(t1[o]) != ACTIVE:
                return False
            Wfe = self._eff(s, "Wf")
            if m.kind == "gap_mean":
                s.params["bf"] += p["g"] * Wfe[:, o]
            elif m.kind == "gap_merge":
                o2 = p["o2"]
                if int(t1[o2]) != ACTIVE:
                    return False
                s.params["Wf"][:, o2] = Wfe[:, o2] + p["alpha"] * Wfe[:, o]
                s.masks["Wf"][:, o2] |= s.masks["Wf"][:, o]
                s.params["bf"] += p["beta"] * Wfe[:, o]
            else:
                raise ValueError(m.kind)
            self._kill1(s, o)
            return True
        raise ValueError(m.site)

    @torch.no_grad()
    def classical_unit_order(self, s):
        X = self.data["train"][0][: self.stats_n]
        P = self.effective(s)
        _, a1, g = self.logits(P, s.struct, X, return_acts=True)
        items = []
        for j in range(self.C1):
            if int(s.struct["type0"][j]) != REMOVED:
                items.append((float(a1[:, j].var() * P["W2"][:, j].pow(2).sum()), ("c1", j)))
        for o in range(self.C2):
            if int(s.struct["type1"][o]) != REMOVED:
                items.append((float(g[:, o].var() * P["Wf"][:, o].pow(2).sum()), ("c2", o)))
        return [u for _, u in sorted(items, key=lambda t: t[0])]

    @torch.no_grad()
    def mean_move(self, s, unit):
        X = self.data["train"][0][: self.stats_n]
        _, a1, g = self.logits(self.effective(s), s.struct, X, return_acts=True)
        site, k = unit
        if site == "c1":
            return Move("chan_mean_in", "c1", 0, {"j": k, "c": float(a1[:, k].mean())})
        return Move("gap_mean", "c2", 1, {"o": k, "g": float(g[:, k].mean())})


def train_epochs(adapter, fit_data, ckpt, seed=0, epochs=10, lr=2e-3, batch=128, deadline=None):
    """Mini-batch Adam training, resumable per epoch. Returns the trained State, or None if
    the deadline was reached first (progress saved in ckpt)."""
    import time
    if os.path.exists(ckpt):
        C = torch.load(ckpt, weights_only=False)
        P, ep0 = C["P"], C["epoch"]
    else:
        P, ep0, C = adapter.init_params(seed), 0, None
    params = [v.requires_grad_(True) for k, v in P.items() if k != "B2"]
    opt = torch.optim.Adam(params, lr=lr)
    if C is not None:
        opt.load_state_dict(C["opt"])
    s = adapter.make_state(P)
    X, y = fit_data
    for ep in range(ep0, epochs):
        if deadline is not None and time.time() > deadline:
            torch.save({"P": {k: v.detach() for k, v in P.items()}, "epoch": ep, "opt": opt.state_dict()}, ckpt)
            return None
        g = torch.Generator().manual_seed(1000 * seed + ep)
        perm = torch.randperm(len(y), generator=g)
        for b in range(0, len(y), batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            F.cross_entropy(adapter.logits(P, s.struct, X[idx]), y[idx]).backward()
            opt.step()
        torch.save({"P": {k: v.detach() for k, v in P.items()}, "epoch": ep + 1, "opt": opt.state_dict()}, ckpt)
    return adapter.make_state({k: v.detach() for k, v in P.items()})
