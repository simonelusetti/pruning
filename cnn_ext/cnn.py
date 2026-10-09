"""
CNN adapter for Veronica's nnprune framework (../NNpruning/nnprune).

Model (the small digits CNN used in walkthrough/):
    x (1x8x8) -> conv1 1->C1, 3x3, zero padding 1 -> ReLU -> a1 (C1 slices)
              -> conv2 C1->C2, 3x3, zero padding 1, + bias b2 + region bias B2 -> ReLU -> a2
              -> global average pool -> g (C2) -> fc C2->10

B2 is a "border-aware" bias for conv2: 9 numbers per conv2 filter (4 corners, 4
edges, interior), expanded to the 8x8 map.  It starts masked (not counted in
k_eff) and becomes live, for the whole layer at once, the first time a move
needs it.  It is the CNN analogue of B_attn in her transformer adapter.

Moves (all realised exactly in the network's own parameters, nothing retrained)
--------------------------------------------------------------------------------
conv1 channel j  (site 'c1', layer 0) -- the move deletes W1[j], b1[j], W2[:, j]:
  chan_mean_in   naive port of her unit_mean: slice -> its mean c, b2 += c * sum(W2[:, j])
  chan_merge_pw  naive port of her unit_merge: a_j ~ alpha a_i + beta pixelwise,
                 W2[:, i] += alpha W2[:, j], b2 += beta * sum(W2[:, j])
  chan_mean_s    output-side constant: b2[o] += mean over data and pixels of channel j's
                 contribution to conv2 filter o
  chan_mean_r    output-side constant per border region: B2[o, region] += region mean
                 of the contribution (activates B2)
  chan_merge_sp  output-side spatial merge: per conv2 filter o, least squares
                 contribution_o(j) ~ V_o * a_i + c_o over the 3x3 neighbourhood of a_i;
                 W2[o, i] += V_o, b2[o] += c_o
conv2 channel o  (site 'c2', layer 1) -- feeds the classifier only through g_o:
  gap_mean       g_o -> its mean, bf += mean * Wf[:, o]                       (exact fold)
  gap_merge      g_o ~ alpha g_o' + beta, Wf[:, o'] += alpha Wf[:, o], bf += beta Wf[:, o]
Single-weight 'zero' moves on W1, W2, Wf come from her core.

Cascades (canonicalize), applied until nothing changes:
  conv1 channel with no live fan-out -> removed with its fan-in
  conv1 channel with no live fan-in  -> constant relu(b1), folded exactly (border-aware) and removed
  conv2 channel with no live fan-out -> removed with its fan-in
  conv2 channel with no live fan-in  -> constant pooled value, folded into bf and removed
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "NNpruning"))

from nnprune.core import ACTIVE, REMOVED, Adapter, Move, State, zero_entries  # noqa: E402

H = W_ = 8
_REG = torch.tensor([0, 1, 1, 1, 1, 1, 1, 2])           # border region index along one axis
REP = [0, 3, 7]                                         # one representative pixel per region

NAIVE_KINDS = ("chan_mean_in", "chan_merge_pw", "gap_mean", "gap_merge")
CNN_KINDS = NAIVE_KINDS + ("chan_mean_s", "chan_mean_r", "chan_merge_sp")


# --------------------------------------------------------------------------
# data and training
# --------------------------------------------------------------------------
def make_digits_data(seed=0):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    t = lambda a: torch.tensor(a)
    return {"train": (t(Xtr), t(ytr).long()), "val": (t(Xva), t(yva).long()),
            "test": (t(Xte), t(yte).long())}


def region_pattern(K: torch.Tensor) -> torch.Tensor:
    """Response of kernels K (..., 3, 3) to a map of ones with zero padding, at the
    9 border regions: (..., 3, 3).  Region (r, c) = sum of the taps that land inside."""
    lead = K.shape[:-2]
    k = K.reshape(-1, 1, 3, 3)
    resp = F.conv2d(torch.ones(1, 1, H, W_), k, padding=1)[0]          # (m, 8, 8)
    return resp[:, REP][:, :, REP].reshape(*lead, 3, 3)


class CNNAdapter(Adapter):
    loss_type = "categorical"

    def __init__(self, data, C1=16, C2=32, n_classes=10):
        super().__init__(data)
        self.C1, self.C2, self.K = C1, C2, n_classes
        self.prunable = ["W1", "W2", "Wf"]
        self.site_layer = {"W1": 0, "W2": 1, "Wf": 2}

    # ---- construction ---------------------------------------------------------
    def init_params(self, seed=0) -> dict:
        g = torch.Generator().manual_seed(seed)
        C1, C2 = self.C1, self.C2
        return {"W1": torch.randn(C1, 1, 3, 3, generator=g) * 0.5, "b1": torch.zeros(C1),
                "W2": torch.randn(C2, C1, 3, 3, generator=g) * (2 / (C1 * 9)) ** 0.5,
                "b2": torch.zeros(C2), "B2": torch.zeros(C2, 3, 3),
                "Wf": torch.randn(self.K, C2, generator=g) * 0.2, "bf": torch.zeros(self.K)}

    def make_state(self, params: dict) -> State:
        params = {k: v.detach().clone() for k, v in params.items()}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in params.items()}
        masks["B2"][:] = False                       # region bias not live until needed
        params["B2"].zero_()
        struct = {"type0": torch.full((self.C1,), ACTIVE, dtype=torch.int8),
                  "type1": torch.full((self.C2,), ACTIVE, dtype=torch.int8)}
        return State(params, masks, struct)

    # ---- forward ----------------------------------------------------------------
    def logits(self, P, S, X, return_acts=False):
        a1 = F.relu(F.conv2d(X, P["W1"], P["b1"], padding=1))
        a1 = a1 * (S["type0"] != REMOVED).to(a1.dtype)[None, :, None, None]
        z2 = F.conv2d(a1, P["W2"], P["b2"], padding=1) + P["B2"][:, _REG][:, :, _REG]
        a2 = F.relu(z2) * (S["type1"] != REMOVED).to(z2.dtype)[None, :, None, None]
        g = a2.mean((2, 3))
        out = g @ P["Wf"].T + P["bf"]
        return (out, a1, g) if return_acts else out

    # ---- bookkeeping ------------------------------------------------------------------
    def _activate_B2(self, s: State) -> None:
        alive = s.struct["type1"] != REMOVED
        s.masks["B2"][alive] = True

    def _kill0(self, s: State, j: int) -> None:
        zero_entries(s, "W1", torch.arange(9) + j * 9)
        zero_entries(s, "b1", torch.tensor([j]))
        col = torch.zeros_like(s.masks["W2"])
        col[:, j] = True
        zero_entries(s, "W2", col)
        s.struct["type0"][j] = REMOVED

    def _kill1(self, s: State, o: int) -> None:
        row = torch.zeros_like(s.masks["W2"])
        row[o] = True
        zero_entries(s, "W2", row)
        zero_entries(s, "b2", torch.tensor([o]))
        rowB = torch.zeros_like(s.masks["B2"])
        rowB[o] = True
        zero_entries(s, "B2", rowB)
        colf = torch.zeros_like(s.masks["Wf"])
        colf[:, o] = True
        zero_entries(s, "Wf", colf)
        s.struct["type1"][o] = REMOVED

    def _eff(self, s: State, name: str) -> torch.Tensor:
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
                    b = float(self._eff(s, "b1")[j])
                    c = max(b, 0.0)
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
                    z = self._eff(s, "b2")[o] + self._eff(s, "B2")[o][_REG][:, _REG]   # (8, 8)
                    gval = float(F.relu(z).mean())
                    s.params["bf"] += gval * self._eff(s, "Wf")[:, o]
                    self._kill1(s, o); changed = True

    # ---- statistics for the structural moves ----------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s: State, cfg) -> dict:
        X, _ = self.data["train"]
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
        # contributions of every conv1 channel to every conv2 filter: (C1, n, C2, 8, 8) -> stats only
        mean_s, mean_r, contrib_flat = {}, {}, {}
        for j in (alive0 if need_contrib else []):
            Cj = F.conv2d(a1[:, j:j + 1], W2[:, j:j + 1], None, padding=1)      # (n, C2, 8, 8)
            mean_s[j] = Cj.mean((0, 2, 3))
            m = Cj.mean(0)                                                    # (C2, 8, 8)
            R = torch.zeros(self.C2, 3, 3)
            for r in range(3):
                for c in range(3):
                    R[:, r, c] = m[:, _REG == r][:, :, _REG == c].mean((1, 2))
            mean_r[j] = R
            contrib_flat[j] = Cj.permute(1, 0, 2, 3).reshape(self.C2, -1)      # (C2, n*64)
        ctx["mean_s"], ctx["mean_r"] = mean_s, mean_r
        ctx["c_in"] = {j: float(a1[:, j].mean()) for j in alive0}
        # pointwise pairs (pixel correlation), as in her MLP adapter
        flat = a1.permute(1, 0, 2, 3).reshape(self.C1, -1)
        pairs = []
        if len(alive0) >= 2 and cfg.n_merge > 0:
            A = flat[alive0]
            Ac = A - A.mean(1, keepdim=True)
            sd = Ac.pow(2).mean(1).sqrt().clamp_min(1e-12)
            Cm = (Ac @ Ac.T) / Ac.shape[1] / (sd[:, None] * sd[None, :])
            iu = torch.triu_indices(len(alive0), len(alive0), 1)
            vals = Cm[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                p_, q_ = int(iu[0, t_]), int(iu[1, t_])
                for i, i2 in ((p_, q_), (q_, p_)):
                    xi, xj = Ac[i], Ac[i2]
                    alpha = float((xi * xj).mean() / xj.pow(2).mean().clamp_min(1e-12))
                    beta = float(A[i].mean() - alpha * A[i2].mean())
                    pairs.append((alive0[i], alive0[i2], alpha, beta))
        ctx["pw_pairs"] = pairs
        # spatial merges: for each removed channel j, best partner by output-side R^2
        patches = F.unfold(a1, 3, padding=1).view(n, self.C1, 9, H * W_)
        patches = patches.permute(1, 0, 3, 2).reshape(self.C1, -1, 9)          # (C1, n*64, 9)
        ones = torch.ones(patches.shape[1], 1)
        gram, Zs = {}, {}
        for i in (alive0 if "chan_merge_sp" in kinds else []):
            Z = torch.cat([patches[i], ones], 1).double()
            Zs[i] = Z
            gram[i] = Z.T @ Z + 1e-8 * torch.eye(10, dtype=torch.float64)
        sp = []
        for j in (alive0 if "chan_merge_sp" in kinds else []):
            T = contrib_flat[j].T.double()                                       # (n*64, C2)
            tot = float((T - T.mean(0)).pow(2).sum())
            best = None
            for i in alive0:
                if i == j:
                    continue
                sol = torch.linalg.solve(gram[i], Zs[i].T @ T)                    # (10, C2)
                rss = float((T - Zs[i] @ sol).pow(2).sum())
                r2 = 1 - rss / max(tot, 1e-12)
                if best is None or r2 > best[0]:
                    best = (r2, i, sol)
            if best is not None:
                r2, i, sol = best
                sp.append((j, i, sol[:9].T.float().reshape(self.C2, 3, 3), sol[9].float(), r2))
        ctx["sp"] = sp
        # conv2 channels (pooled)
        ctx["g_mean"] = g.mean(0)
        gp = []
        if len(alive1) >= 2 and cfg.n_merge > 0:
            G = g[:, alive1].T
            Gc = G - G.mean(1, keepdim=True)
            sd = Gc.pow(2).mean(1).sqrt().clamp_min(1e-12)
            Cg = (Gc @ Gc.T) / Gc.shape[1] / (sd[:, None] * sd[None, :])
            iu = torch.triu_indices(len(alive1), len(alive1), 1)
            vals = Cg[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                p_, q_ = int(iu[0, t_]), int(iu[1, t_])
                for o, o2 in ((p_, q_), (q_, p_)):
                    xo, x2 = Gc[o], Gc[o2]
                    alpha = float((xo * x2).mean() / x2.pow(2).mean().clamp_min(1e-12))
                    beta = float(G[o].mean() - alpha * G[o2].mean())
                    gp.append((alive1[o], alive1[o2], alpha, beta))
        ctx["gap_pairs"] = gp
        return ctx

    def structural_moves(self, s: State, ctx: dict, cfg) -> list:
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

    def apply_structural(self, s: State, m: Move) -> bool:
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
            elif m.kind == "chan_merge_pw":
                i = p["i"]
                if int(t0[i]) != ACTIVE:
                    return False
                s.params["W2"][:, i] = W2e[:, i] + p["alpha"] * W2e[:, j]
                s.masks["W2"][:, i] |= s.masks["W2"][:, j]
                s.params["b2"] += p["beta"] * W2e[:, j].sum((1, 2))
            elif m.kind == "chan_merge_sp":
                i = p["i"]
                if int(t0[i]) != ACTIVE:
                    return False
                alive_o = (t1 != REMOVED)
                s.params["W2"][:, i] = W2e[:, i] + p["V"]
                s.masks["W2"][alive_o, i] = True
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

    # ---- classical structured baseline (mean ablation, naive input-side fold) ----------------
    @torch.no_grad()
    def classical_unit_order(self, s: State) -> list:
        X, _ = self.data["train"]
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
    def mean_move(self, s: State, unit) -> Move:
        X, _ = self.data["train"]
        _, a1, g = self.logits(self.effective(s), s.struct, X, return_acts=True)
        site, k = unit
        if site == "c1":
            return Move("chan_mean_in", "c1", 0, {"j": k, "c": float(a1[:, k].mean())})
        return Move("gap_mean", "c2", 1, {"o": k, "g": float(g[:, k].mean())})


def train_cnn(adapter: CNNAdapter, seed=0, steps=1500, lr=3e-3):
    torch.manual_seed(seed)
    P = {k: v.requires_grad_(k != "B2") for k, v in adapter.init_params(seed).items()}
    s = adapter.make_state(P)
    X, y = adapter.data["train"]
    opt = torch.optim.Adam([v for k, v in P.items() if k != "B2"], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(adapter.logits(P, s.struct, X), y).backward()
        opt.step()
    return adapter.make_state(P)
