"""
Stream-direction move for Veronica's one-layer transformer (extends
nnprune.transformer.TransformerAdapter; her code is not modified).

The stream x in R^d (d = 64) is changed only by additions (E, P, W_O, B_attn, W_out,
b_out) and read only through matrices (W_Q, W_K, W_V at depth 0, W_in at depth 1,
W_U at depth 2).  So x -> H x, with every added term multiplied by H and every
reading matrix by H^{-1}, leaves the function unchanged for any invertible H.

Move `stream_mean` (payload: a unit vector u, stored in the ORIGINAL coordinates):
  1. reflect the live coordinates with the Householder matrix H that sends u to the
     last live coordinate j (exact; H is symmetric and orthogonal);
  2. replace x_j by its training mean, depth by depth, and fold the constants:
       depth 0 (per position t):  q, k, v biases  bQ/bK/bV[h, t] += m0_t W_{Q/K/V}[h, j, :]
       depth 1 (query position):  b_in += m1 W_in[:, j]
       depth 2 (query position):  b_U  += m2 W_U[j, :]
     (each mean is computed after the previous depth was replaced);
  3. delete coordinate j everywhere: column j of W_E, W_pos, W_O, B_attn, entry j of
     b_out, row j of W_out, and the reading entries of W_Q, W_K, W_V, W_in, W_U.
  Freed on the dense model: 891 parameters; the new biases bQ (query position only),
  bK, bV cost 64 + 192 + 192 = 448 the first time they are used (counted once).
  After step 1 the stream-indexed tensors are dense on the live coordinates: weights
  zeroed earlier by single-weight moves come back, and k_eff counts them.

Candidates (refresh): smallest eigen-directions of the per-position-centred stream
covariances within the live coordinates: 4 of the pooled sum_k C_k / tr C_k and 2 of
each C_k (x0 at positions 0 and 1; x1 and x2 at the query position).

Bookkeeping: struct['R'] is the accumulated rotation (x_now = R x_orig) and
struct['stream_alive'] the live coordinates.  Statistics computed at a refresh in the
basis of that moment (Cq, Ck, Cy, head means) are rotated to the current basis when
a qk_rank / ov_rank / head_mean move is applied after a stream move.
Cascade: a live coordinate with no live reading entry loses its added terms, and one
with no live added term (it is identically 0) loses its reading entries.

The q/k/v biases live in each head's d_h-dimensional space.  Her qk_rank / ov_rank
moves change the basis of that space, so when a head has live biases the rank cut is
done on the augmented input [x_t, e_t] (e_t = one-hot of the position), i.e. on the
stacked matrices [W; b], with the same activation-weighted SVD; the head canonicalisation
treats a head dimension as live if its weight column OR its bias column is live.
Dead stream coordinates are re-masked after every move (her rank cuts set whole
columns live).
"""
from __future__ import annotations

import math

import torch

from nnprune import transformer as T
from nnprune.core import Move, zero_entries

STREAM_KINDS = ("stream_mean",)
# stream-indexed tensors: name -> axis of the stream index
ADDED = {"W_E": 1, "W_pos": 1, "W_O": 2, "B_attn": 1, "b_out": 0, "W_out": 0}
READ = {"W_Q": 1, "W_K": 1, "W_V": 1, "W_in": 1, "W_U": 0}


class StreamAdapter(T.TransformerAdapter):

    def __init__(self, data, p, n_pool=4, n_each=2, fresh=False, **kw):
        """fresh=True: a stream candidate is a slot ('the i-th direction of the current
        candidate list'), recomputed from the current network when the move is applied,
        so candidates never go stale between refreshes.  fresh=False: fixed directions
        from the last refresh (the first version)."""
        super().__init__(data, p, **kw)
        self.n_pool, self.n_each, self.fresh = n_pool, n_each, fresh

    # ---- construction ------------------------------------------------------------
    def init_params(self, seed=0):
        P = super().init_params(seed)
        for n in ("bQ", "bK", "bV"):
            P[n] = torch.zeros(self.H, self.T, self.dh)
        return P

    def make_state(self, params):
        params = dict(params)
        for n in ("bQ", "bK", "bV"):
            params.setdefault(n, torch.zeros(self.H, self.T, self.dh))
        s = super().make_state(params)
        for n in ("bQ", "bK", "bV"):
            s.masks[n][:] = False
        s.struct["stream_alive"] = torch.ones(self.d, dtype=torch.bool)
        s.struct["R"] = torch.eye(self.d)
        return s

    def upgrade(self, s):
        """State saved by her TransformerAdapter -> state of this adapter (same function)."""
        if "bQ" in s.params:
            return s
        s = s.clone()
        for n in ("bQ", "bK", "bV"):
            s.params[n] = torch.zeros(self.H, self.T, self.dh)
            s.masks[n] = torch.zeros(self.H, self.T, self.dh, dtype=torch.bool)
        s.struct["stream_alive"] = torch.ones(self.d, dtype=torch.bool)
        s.struct["R"] = torch.eye(self.d)
        return s

    # ---- forward: hers, plus the q/k/v biases ---------------------------------------
    def forward_full(self, P, S, X):
        x0 = P["W_E"][X] + P["W_pos"]
        q = torch.einsum("btd,hde->bhte", x0, P["W_Q"]) + P["bQ"]
        k = torch.einsum("btd,hde->bhte", x0, P["W_K"]) + P["bK"]
        v = torch.einsum("btd,hde->bhte", x0, P["W_V"]) + P["bV"]
        sc = torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(self.dh)
        sc = sc.masked_fill(~self.causal, float("-inf"))
        attn = torch.softmax(sc, -1)
        attn = torch.where(S["head_fixed"][:, None, None], S["A_fixed"], attn)
        z = torch.einsum("bhts,bhse->bhte", attn, v)
        ho = torch.einsum("bhte,hed->bhtd", z, P["W_O"])
        ho = ho * S["head_alive"].to(ho.dtype)[:, None, None]
        x1 = x0 + ho.sum(-3) + P["B_attn"]
        act = torch.relu(x1 @ P["W_in"].T + P["b_in"])
        act = act * (S["unit_type"] == T.ACTIVE).to(act.dtype)
        x2 = x1 + act @ P["W_out"].T + P["b_out"]
        logits = x2[..., -1, :] @ P["W_U"] + P["b_U"]
        return logits, {"x0": x0, "attn": attn, "z": z, "ho": ho, "x1": x1, "act": act, "x2": x2}

    # ---- the exact reflection ------------------------------------------------------------
    @staticmethod
    def _householder(u, j, d):
        e = torch.zeros(d, dtype=u.dtype); e[j] = 1
        w = u - e
        nw = float(w.norm())
        if nw < 1e-9:
            return torch.eye(d)
        w = w / nw
        return (torch.eye(d, dtype=u.dtype) - 2 * torch.outer(w, w)).float()

    def _rotate(self, s, H):
        """x -> H x on the whole stream (H acts as the identity on dead coordinates)."""
        alive = s.struct["stream_alive"]
        P = self.effective(s)
        new = {
            "W_E": P["W_E"] @ H, "W_pos": P["W_pos"] @ H, "B_attn": P["B_attn"] @ H,
            "W_O": P["W_O"] @ H, "b_out": H @ P["b_out"], "W_out": H @ P["W_out"],
            "W_Q": torch.einsum("dk,hke->hde", H, P["W_Q"]),
            "W_K": torch.einsum("dk,hke->hde", H, P["W_K"]),
            "W_V": torch.einsum("dk,hke->hde", H, P["W_V"]),
            "W_in": P["W_in"] @ H, "W_U": H @ P["W_U"],
        }
        for name, val in new.items():
            ax = ADDED.get(name, READ.get(name))
            m = s.masks[name]
            live_other = m.any(dim=ax, keepdim=True)
            shape = [1] * m.dim(); shape[ax] = -1
            nm = live_other & alive.view(shape)
            s.masks[name] = nm.contiguous()
            s.params[name] = (val * nm.to(val.dtype)).contiguous()
        s.struct["R"] = H @ s.struct["R"]

    # ---- the move ---------------------------------------------------------------------------
    @torch.no_grad()
    def _stream_mean(self, s, u_orig):
        alive = s.struct["stream_alive"]
        if int(alive.sum()) <= 1:
            return False
        u = (s.struct["R"] @ u_orig.float()).double()
        u[~alive] = 0
        n = float(u.norm())
        if n < 0.5:
            return False
        u = u / n
        j = int(alive.nonzero().max())
        self._rotate(s, self._householder(u, j, self.d))
        X = self.data["train"][0]
        qp = self.qpos[0]
        # depth 0: per-position mean of x0_j, folded into the q/k/v biases
        _, c = self.forward_full(self.effective(s), s.struct, X)
        m0 = c["x0"][:, :, j].mean(0)                                         # (T,)
        for n_, W in (("bQ", "W_Q"), ("bK", "W_K"), ("bV", "W_V")):
            Wj = s.params[W][:, j, :] * s.masks[W][:, j, :].to(s.params[W].dtype)   # (H, dh)
            s.params[n_] += m0[None, :, None] * Wj[:, None, :]
            if n_ == "bQ":
                s.masks[n_][:, qp, :] = True
            else:
                s.masks[n_][:] = True
            mk = torch.zeros_like(s.masks[W]); mk[:, j, :] = True
            zero_entries(s, W, mk)
        # depth 1
        _, c = self.forward_full(self.effective(s), s.struct, X)
        m1 = float(c["x1"][:, qp, j].mean())
        s.params["b_in"] += m1 * s.params["W_in"][:, j] * s.masks["W_in"][:, j].to(s.params["W_in"].dtype)
        mk = torch.zeros_like(s.masks["W_in"]); mk[:, j] = True
        zero_entries(s, "W_in", mk)
        # depth 2
        _, c = self.forward_full(self.effective(s), s.struct, X)
        m2 = float(c["x2"][:, qp, j].mean())
        s.params["b_U"] += m2 * s.params["W_U"][j] * s.masks["W_U"][j].to(s.params["W_U"].dtype)
        mk = torch.zeros_like(s.masks["W_U"]); mk[j] = True
        zero_entries(s, "W_U", mk)
        self._kill_added(s, j)
        return True

    def _kill_added(self, s, j):
        for name, ax in ADDED.items():
            mk = torch.zeros_like(s.masks[name])
            mk.select(ax, j).fill_(True)
            zero_entries(s, name, mk)
        s.struct["stream_alive"][j] = False

    def _kill_read(self, s, j):
        for name, ax in READ.items():
            mk = torch.zeros_like(s.masks[name])
            mk.select(ax, j).fill_(True)
            zero_entries(s, name, mk)

    def canonicalize(self, s):
        self._remask_dead(s)
        self._head_canonicalize(s)
        self._unit_canonicalize(s)
        alive = s.struct["stream_alive"]
        for j in alive.nonzero().flatten().tolist():
            read = any(bool(s.masks[n].select(ax, j).any()) for n, ax in READ.items())
            added = any(bool(s.masks[n].select(ax, j).any()) for n, ax in ADDED.items())
            if not read:
                self._kill_added(s, j)
            elif not added:
                self._kill_read(s, j)
                s.struct["stream_alive"][j] = False
        self._head_canonicalize(s)
        self._unit_canonicalize(s)

    def _unit_canonicalize(self, s):
        """Her MLP-unit cascade (transformer.py canonicalize, second half), unchanged."""
        changed = True
        while changed:
            changed = False
            t = s.struct["unit_type"]
            fan_out = s.masks["W_out"].any(0)
            fan_in = s.masks["W_in"].any(1)
            for i in ((t == T.ACTIVE) & ~fan_out).nonzero().flatten().tolist():
                self._kill_unit(s, i); changed = True
            t = s.struct["unit_type"]
            for i in ((t == T.ACTIVE) & ~fan_in).nonzero().flatten().tolist():
                b = s.params["b_in"][i] if bool(s.masks["b_in"][i]) else torch.tensor(0.0)
                self._fold_unit(s, i, float(torch.relu(b))); changed = True

    # ---- head bookkeeping with the q/k/v biases ------------------------------------------------
    def _kill_head(self, s, h):
        super()._kill_head(s, h)
        for n in ("bQ", "bK", "bV"):
            mk = torch.zeros_like(s.masks[n]); mk[h] = True
            zero_entries(s, n, mk)

    def _zero_qk(self, s, h):
        super()._zero_qk(s, h)
        for n in ("bQ", "bK"):
            mk = torch.zeros_like(s.masks[n]); mk[h] = True
            zero_entries(s, n, mk)

    def _head_canonicalize(self, s):
        """Her head rule, with a head dimension live if its weight OR bias column is live."""
        st = s.struct
        for h in range(self.H):
            if not bool(st["head_alive"][h]):
                if any(bool(s.masks[n][h].any()) for n in ("W_Q", "W_K", "W_V", "W_O", "bQ", "bK", "bV")):
                    self._kill_head(s, h)
                continue
            q = s.masks["W_Q"][h].any(0) | s.masks["bQ"][h].any(0)
            k = s.masks["W_K"][h].any(0) | s.masks["bK"][h].any(0)
            v = s.masks["W_V"][h].any(0) | s.masks["bV"][h].any(0)
            qk, ov = q & k, v & s.masks["W_O"][h].any(1)
            for e in (~qk).nonzero().flatten().tolist():
                for n, ax in (("W_Q", 1), ("W_K", 1), ("bQ", 1), ("bK", 1)):
                    mk = torch.zeros_like(s.masks[n]); mk[h].select(ax, e).fill_(True)
                    zero_entries(s, n, mk)
            for e in (~ov).nonzero().flatten().tolist():
                for n in ("W_V", "bV"):
                    mk = torch.zeros_like(s.masks[n]); mk[h][:, e] = True
                    zero_entries(s, n, mk)
                mk = torch.zeros_like(s.masks["W_O"]); mk[h, e, :] = True
                zero_entries(s, "W_O", mk)
            if not bool(ov.any()):
                self._kill_head(s, h)
            elif bool(st["head_fixed"][h]) and bool(qk.any()):
                self._zero_qk(s, h)

    def _remask_dead(self, s):
        for j in (~s.struct["stream_alive"]).nonzero().flatten().tolist():
            for name, ax in list(ADDED.items()) + list(READ.items()):
                if bool(s.masks[name].select(ax, j).any()):
                    mk = torch.zeros_like(s.masks[name]); mk.select(ax, j).fill_(True)
                    zero_entries(s, name, mk)

    def qk_rank(self, s, h):
        q = s.masks["W_Q"][h].any(0) | s.masks["bQ"][h].any(0)
        k = s.masks["W_K"][h].any(0) | s.masks["bK"][h].any(0)
        return int((q & k).sum())

    def ov_rank(self, s, h):
        v = s.masks["W_V"][h].any(0) | s.masks["bV"][h].any(0)
        return int((v & s.masks["W_O"][h].any(1)).sum())

    @torch.no_grad()
    def _augmented_rank_cut(self, s, kind, h):
        """Rank cut of [W_Q; bQ][W_K; bK]^T (or [W_V; bV] W_O) on inputs [x_t, e_t]."""
        X = self.data["train"][0]
        _, c = self.forward_full(self.effective(s), s.struct, X)
        n, Tn, d = c["x0"].shape
        aug = torch.cat([c["x0"], torch.eye(Tn).expand(n, Tn, Tn)], -1).double()   # (n, T, d+T)
        qp = self.qpos
        E = lambda name: (s.params[name][h] * s.masks[name][h].to(s.params[name].dtype))
        if kind == "qk_rank":
            r = self.qk_rank(s, h) - 1
            if r < 1:
                return False
            WQ = torch.cat([E("W_Q"), E("bQ")], 0)          # (d+T, dh)
            WK = torch.cat([E("W_K"), E("bK")], 0)
            aq = aug[:, qp].reshape(-1, d + Tn); ak = aug.reshape(-1, d + Tn)
            A, B = T.weighted_lowrank(WQ @ WK.T, aq.T @ aq / len(aq), ak.T @ ak / len(ak), r)
            for name, bname, F_ in (("W_Q", "bQ", A), ("W_K", "bK", B)):
                s.params[name][h].zero_(); s.masks[name][h] = False
                s.params[bname][h].zero_(); s.masks[bname][h] = False
                s.params[name][h][:, :r] = F_[:d]; s.masks[name][h][:, :r] = True
                s.params[bname][h][:, :r] = F_[d:]
                if bname == "bQ":
                    s.masks[bname][h][qp, :r] = True
                else:
                    s.masks[bname][h][:, :r] = True
            return True
        r = self.ov_rank(s, h) - 1
        if r < 1:
            return False
        WV = torch.cat([E("W_V"), E("bV")], 0)              # (d+T, dh)
        y = torch.einsum("bts,bsd->btd", c["attn"][:, h, qp].double(), aug).reshape(-1, d + Tn)
        A, B = T.weighted_lowrank(WV @ E("W_O"), y.T @ y / len(y), torch.eye(d), r)
        s.params["W_V"][h].zero_(); s.masks["W_V"][h] = False
        s.params["bV"][h].zero_(); s.masks["bV"][h] = False
        s.params["W_V"][h][:, :r] = A[:d]; s.masks["W_V"][h][:, :r] = True
        s.params["bV"][h][:, :r] = A[d:]; s.masks["bV"][h][:, :r] = True
        s.params["W_O"][h].zero_(); s.masks["W_O"][h] = False
        s.params["W_O"][h][:r, :] = B.T; s.masks["W_O"][h][:r, :] = True
        return True

    # ---- statistics and candidates ------------------------------------------------------------
    @torch.no_grad()
    @torch.no_grad()
    def stream_candidates(self, s):
        """Unit vectors (ORIGINAL coordinates): smallest eigen-directions, within the live
        coordinates, of the pooled covariance (n_pool) and of each depth's (n_each)."""
        if int(s.struct["stream_alive"].sum()) <= 1:
            return []
        X = self.data["train"][0]
        _, c = self.forward_full(self.effective(s), s.struct, X)
        qp = self.qpos[0]
        L = s.struct["stream_alive"].nonzero().flatten()
        x0 = c["x0"][:, :, L]
        a0 = torch.cat([x0[:, t] - x0[:, t].mean(0) for t in range(self.T) if t != qp]).double()
        a1 = (c["x1"][:, qp, L] - c["x1"][:, qp, L].mean(0)).double()
        a2 = (c["x2"][:, qp, L] - c["x2"][:, qp, L].mean(0)).double()
        Cs = [a.T @ a / len(a) for a in (a0, a1, a2)]
        Cs = [C / max(float(torch.trace(C)), 1e-30) for C in Cs]
        vecs = []
        for C, n in [(sum(Cs), self.n_pool)] + [(C, self.n_each) for C in Cs]:
            _, E = torch.linalg.eigh(C)
            for i in range(n):
                v = torch.zeros(self.d, dtype=torch.double)
                if i < E.shape[1]:
                    v[L] = E[:, i]
                vecs.append(v)
        R = s.struct["R"].double()
        return [(R.T @ v).float() for v in vecs]

    @torch.no_grad()
    def refresh_extra(self, s, cfg):
        ctx = super().refresh_extra(s, cfg)
        ctx["R_ref"] = s.struct["R"].clone()
        kinds = set(cfg.struct_kinds) if cfg.struct_kinds is not None else set(STREAM_KINDS)
        ctx["stream_cands"] = []
        if not (kinds & set(STREAM_KINDS)) or int(s.struct["stream_alive"].sum()) <= 1:
            return ctx
        ctx["stream_cands"] = self.stream_candidates(s)
        return ctx

    def structural_moves(self, s, ctx, cfg):
        mv = super().structural_moves(s, ctx, cfg)
        for m in mv:
            if m.kind in ("qk_rank", "ov_rank", "head_mean"):
                m.payload["R_ref"] = ctx["R_ref"]
        for i, u in enumerate(ctx.get("stream_cands", [])):
            mv.append(Move("stream_mean", "stream", 0, {"slot": i} if self.fresh else {"u": u}))
        return mv

    def apply_structural(self, s, m):
        if m.kind == "stream_mean":
            if "slot" in m.payload:
                cands = self.stream_candidates(s)
                if m.payload["slot"] >= len(cands) or float(cands[m.payload["slot"]].norm()) < 0.5:
                    return False
                return self._stream_mean(s, cands[m.payload["slot"]])
            return self._stream_mean(s, m.payload["u"])
        if m.kind in ("qk_rank", "ov_rank"):
            h = m.payload["h"]
            if not bool(s.struct["head_alive"][h]):
                return False
            names = ("bQ", "bK") if m.kind == "qk_rank" else ("bV",)
            if any(bool(s.masks[n][h].any()) for n in names):
                if m.kind == "qk_rank" and (bool(s.struct["head_fixed"][h]) or self.qk_rank(s, h) == 0):
                    return False
                return self._augmented_rank_cut(s, m.kind, h)
        R_ref = m.payload.get("R_ref")
        if R_ref is not None and not torch.equal(R_ref, s.struct["R"]):
            M = s.struct["R"] @ R_ref.T                       # basis of the refresh -> current basis
            p = dict(m.payload)
            for key in ("Cq", "Ck", "Cy"):
                if key in p:
                    p[key] = (M.double() @ p[key].double() @ M.double().T).to(p[key].dtype)
            if "mean" in p:
                p["mean"] = p["mean"] @ M.T
            m = Move(m.kind, m.site, m.layer, p, m.pred_cost)
        return super().apply_structural(s, m)
