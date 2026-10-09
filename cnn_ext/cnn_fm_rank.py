"""
Low-rank convolution move for the Fashion-MNIST CNN (extends cnn_fm.CNNAdapterFM).

conv2 has kernel W2 of shape (C2, C1, 3, 3).  Read as a matrix M (C2 x C1*9), its
output at one pixel is z = M x + b, where x is the 3x3 patch of all C1 input slices
around that pixel (zero padding included).  A rank-r factorisation M ~ B A with
A (r x C1*9) and B (C2 x r) is exactly a 3x3 conv C1 -> r (kernel A) followed by a 1x1
conv r -> C2 (kernel B).  It stores r*(C1*9 + C2) numbers instead of C2*C1*9.

The factorisation is the closest rank-r matrix in the norm that matters downstream,
the expected squared change of conv2's pre-activation over the data:
    min  E || (M - M') x ||^2  =  || (M - M') C^{1/2} ||_F^2,   C = E[x x^T]
solved by the SVD of M C^{1/2} (Eckart-Young), as her qk_rank / ov_rank do for
attention.  The mean change E[(M - M') x] is added to b2, so the average
pre-activation of every conv2 filter is unchanged.

Moves (site 'c2w', layer 1):
  conv_lowrank   dense conv2 -> rank r, for r in RANKS (only ranks that save parameters)
  conv_rankcut   factorised conv2 of rank r -> rank r-1 (and r-4)
Once conv2 is factorised:
  - pointwise merges act on the 3x3 factor A (column operation, stays rank r)
  - the spatial merge is not offered (it would add a full-rank correction)
  - single-weight zero moves act on the entries of A and B
  - cascades also remove a rank component whose row of A or column of B is dead

Representation: params W2 (dense), W2a (C2_max x C1 x 3 x 3) = A, W2b (C2 x C2_max) = B.
Exactly one of {W2} or {W2a, W2b} is live (masks), struct['fact'] says which.
`effective()` folds B A into W2, so every statistic of the parent class sees the
dense kernel; `logits` adds B A to W2, so it is correct for raw parameters too.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cnn_fm import CNN_KINDS, HM, CNNAdapterFM, region_pattern, _REG
from nnprune.core import ACTIVE, REMOVED, Move, zero_entries

RANK_KINDS = ("conv_lowrank", "conv_rankcut")
RANKS = (24, 20, 16, 12, 8, 6, 4)


class CNNAdapterFMRank(CNNAdapterFM):

    def __init__(self, data, **kw):
        super().__init__(data, **kw)
        self.prunable = ["W1", "W2", "W2a", "W2b", "Wf"]
        self.site_layer = {"W1": 0, "W2": 1, "W2a": 1, "W2b": 1, "Wf": 2}

    # ---- construction ----------------------------------------------------------------
    def init_params(self, seed=0):
        P = super().init_params(seed)
        P["W2a"] = torch.zeros(self.C2, self.C1, 3, 3)
        P["W2b"] = torch.zeros(self.C2, self.C2)
        return P

    def make_state(self, params):
        params = dict(params)
        params.setdefault("W2a", torch.zeros(self.C2, self.C1, 3, 3))
        params.setdefault("W2b", torch.zeros(self.C2, self.C2))
        s = super().make_state(params)
        s.masks["W2a"][:] = False
        s.masks["W2b"][:] = False
        s.struct["fact"] = torch.zeros(1, dtype=torch.int8)
        return s

    def upgrade(self, s):
        """Turn a State saved by cnn_fm (no factor tensors) into one for this adapter."""
        if "W2a" in s.params:
            return s
        s = s.clone()
        s.params["W2a"] = torch.zeros(self.C2, self.C1, 3, 3)
        s.params["W2b"] = torch.zeros(self.C2, self.C2)
        s.masks["W2a"] = torch.zeros(self.C2, self.C1, 3, 3, dtype=torch.bool)
        s.masks["W2b"] = torch.zeros(self.C2, self.C2, dtype=torch.bool)
        s.struct["fact"] = torch.zeros(1, dtype=torch.int8)
        return s

    # ---- forward ---------------------------------------------------------------------
    def _fold(self, W2, W2a, W2b):
        return W2 + (W2b @ W2a.reshape(W2a.shape[0], -1)).reshape(W2.shape)

    def logits(self, P, S, X, return_acts=False):
        Q = dict(P)
        Q["W2"] = self._fold(P["W2"], P["W2a"], P["W2b"])
        return super().logits(Q, S, X, return_acts)

    def effective(self, s):
        P = super().effective(s)
        if int(s.struct["fact"][0]):
            P["W2"] = self._fold(P["W2"], P["W2a"], P["W2b"])
            P["W2a"] = torch.zeros_like(P["W2a"])
            P["W2b"] = torch.zeros_like(P["W2b"])
        return P

    def _eff(self, s, name):
        e = s.params[name] * s.masks[name].to(s.params[name].dtype)
        if name == "W2" and int(s.struct["fact"][0]):
            e = self._fold(e, self._eff_raw(s, "W2a"), self._eff_raw(s, "W2b"))
        return e

    def _eff_raw(self, s, name):
        return s.params[name] * s.masks[name].to(s.params[name].dtype)

    def _fact(self, s):
        return bool(int(s.struct["fact"][0]))

    # ---- bookkeeping -------------------------------------------------------------------
    def _kill0(self, s, j):
        super()._kill0(s, j)
        col = torch.zeros_like(s.masks["W2a"]); col[:, j] = True
        zero_entries(s, "W2a", col)

    def _kill1(self, s, o):
        super()._kill1(s, o)
        row = torch.zeros_like(s.masks["W2b"]); row[o] = True
        zero_entries(s, "W2b", row)

    def _kill_comp(self, s, k):
        row = torch.zeros_like(s.masks["W2a"]); row[k] = True
        zero_entries(s, "W2a", row)
        col = torch.zeros_like(s.masks["W2b"]); col[:, k] = True
        zero_entries(s, "W2b", col)

    def canonicalize(self, s):
        if not self._fact(s):
            return super().canonicalize(s)
        changed = True
        while changed:
            changed = False
            t0, t1 = s.struct["type0"], s.struct["type1"]
            for k in range(self.C2):                      # dead rank components
                a_live, b_live = bool(s.masks["W2a"][k].any()), bool(s.masks["W2b"][:, k].any())
                if a_live != b_live:
                    self._kill_comp(s, k); changed = True
            for j in range(self.C1):
                if int(t0[j]) == REMOVED:
                    continue
                if not bool(s.masks["W2a"][:, j].any()):
                    self._kill0(s, j); changed = True; continue
                if not bool(s.masks["W1"][j].any()):
                    c = max(float(self._eff_raw(s, "b1")[j]), 0.0)
                    if c != 0.0:
                        self._activate_B2(s)
                        s.params["B2"] += c * region_pattern(self._eff(s, "W2")[:, j])
                    self._kill0(s, j); changed = True
            for o in range(self.C2):
                if int(t1[o]) == REMOVED:
                    continue
                if not bool(s.masks["Wf"][:, o].any()):
                    self._kill1(s, o); changed = True; continue
                if not bool(s.masks["W2b"][o].any()):
                    z = self._eff_raw(s, "b2")[o] + self._eff_raw(s, "B2")[o][_REG][:, _REG]
                    s.params["bf"] += float(F.relu(z).mean()) * self._eff_raw(s, "Wf")[:, o]
                    self._kill1(s, o); changed = True

    # ---- statistics ----------------------------------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s, cfg):
        ctx = super().refresh_extra(s, cfg)
        kinds = set(cfg.struct_kinds) if cfg.struct_kinds is not None else set(CNN_KINDS + RANK_KINDS)
        if not cfg.use_struct or not (kinds & set(RANK_KINDS)):
            ctx["rank"] = []
            return ctx
        alive0 = s.struct["type0"] != REMOVED
        alive1 = s.struct["type1"] != REMOVED
        n_in, n_out = int(alive0.sum()) * 9, int(alive1.sum())
        if self._fact(s):
            r_cur = int((s.masks["W2a"].reshape(self.C2, -1).any(1)).sum())
            targets = [r for r in (r_cur - 1, r_cur - 4) if r >= 1]
        else:
            targets = [r for r in RANKS if r * (n_in + n_out) < int(s.masks["W2"].sum())]
        ctx["rank"] = self._rank_moves(s, targets)
        return ctx

    @torch.no_grad()
    def _rank_moves(self, s, targets):
        """Activation-weighted rank-r factorisations of the current conv2 kernel."""
        X = self.data["train"][0][: self.stats_n]
        P = self.effective(s)
        a1 = self.conv1_out(P, s.struct, X)
        n = a1.shape[0]
        x = F.unfold(a1, 3, padding=1).permute(0, 2, 1).reshape(n * HM * HM, -1).double()   # patches (N, C1*9)
        mu = x.mean(0)
        C = (x.T @ x) / x.shape[0]
        lam, Q = torch.linalg.eigh(C)
        lam = lam.clamp_min(1e-8 * float(lam.max()))
        Ch, Cih = (Q * lam.sqrt()) @ Q.T, (Q * lam.rsqrt()) @ Q.T
        M = P["W2"].reshape(self.C2, -1).double()
        U, S, Vt = torch.linalg.svd(M @ Ch, full_matrices=False)
        moves = []
        for r in targets:
            A = (S[:r, None] * Vt[:r]) @ Cih                     # (r, C1*9)
            B = U[:, :r]                                        # (C2, r)
            db = (M - B @ A) @ mu                               # mean change of the pre-activation
            moves.append({"r": r, "A": A.float().reshape(r, self.C1, 3, 3), "B": B.float(), "db": db.float()})
        return moves

    def structural_moves(self, s, ctx, cfg):
        mv = super().structural_moves(s, ctx, cfg)
        if self._fact(s):
            mv = [m for m in mv if m.kind != "chan_merge_sp"]
        kind = "conv_rankcut" if self._fact(s) else "conv_lowrank"
        for p in ctx.get("rank", []):
            mv.append(Move(kind, "c2w", 1, p))
        return mv

    def apply_structural(self, s, m):
        if m.kind in RANK_KINDS:
            p = m.payload
            if (m.kind == "conv_rankcut") != self._fact(s):
                return False
            r = p["r"]
            alive0 = s.struct["type0"] != REMOVED
            alive1 = s.struct["type1"] != REMOVED
            zero_entries(s, "W2", torch.ones_like(s.masks["W2"]))
            s.params["W2a"].zero_(); s.masks["W2a"][:] = False
            s.params["W2b"].zero_(); s.masks["W2b"][:] = False
            s.params["W2a"][:r] = p["A"]
            s.masks["W2a"][:r] = alive0[None, :, None, None].expand(r, -1, 3, 3)
            s.params["W2b"][:, :r] = p["B"]
            s.masks["W2b"][:, :r] = alive1[:, None].expand(-1, r)
            s.params["b2"] += p["db"]
            s.struct["fact"][0] = 1
            return True
        if self._fact(s) and m.kind == "chan_merge_sp":
            return False
        if self._fact(s) and m.kind == "chan_merge_pw":
            p = m.payload
            j, i = p["j"], p["i"]
            t0 = s.struct["type0"]
            if int(t0[j]) != ACTIVE or int(t0[i]) != ACTIVE:
                return False
            W2e = self._eff(s, "W2")
            Ae = self._eff_raw(s, "W2a")
            s.params["W2a"][:, i] = Ae[:, i] + p["alpha"] * Ae[:, j]
            s.masks["W2a"][:, i] |= s.masks["W2a"][:, j]
            s.params["b2"] += p["beta"] * W2e[:, j].sum((1, 2))
            self._kill0(s, j)
            return True
        return super().apply_structural(s, m)
