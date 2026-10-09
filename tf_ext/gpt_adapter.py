"""
Veronica's pruning framework (nnprune.core) on the toy GPT of gpt_toy.py: L pre-LN
blocks, GELU, all biases, untied unembedding, character LM on Tiny Shakespeare.
Her code is not modified; this file is an Adapter for her core.

Samples and splits
------------------
A sample is one window of T = 32 characters; its loss is the mean next-character loss
over the 32 positions.  'train' = 2048 fixed training windows (screen, refresh),
'stats' = the first 512 of them (activation statistics of the moves), 'val' = all
3485 validation windows, halved at random: 'valB' (stops, 1743 windows) is one half,
'valA' (chooses, 512 windows = 16 384 scored characters) is taken from the other half,
'test' = all test windows (never used for decisions).

Stage 1: exact folds (no statistic; checked on the logits)
-----------------------------------------------------------
With C = I - 11^T/d (d = 64), applied to every trained LayerNorm network:
  * every term added to the stream is multiplied by C (W_E, W_pos, W_O, b_O, W_2, b_2):
    LN(Cx) = LN(x), and afterwards x is orthogonal to 1 at every depth;
  * gamma, beta of every LayerNorm go into the matrix that reads it:
    W (gamma * n + beta) + b = (W diag gamma) n + (W beta + b);
  * b_K (and the beta part of the keys) is deleted: q.(k_s + c) - q.k_s does not
    depend on s, and the softmax over s ignores it;
  * b_V (with its beta part) is folded into b_O: attention rows sum to 1;
  * the direction 1/sqrt(d) is identically 0 on the stream: the basis is rotated so
    that it is a coordinate, and the coordinate is deleted (writers and readers).
Each LayerNorm then reads  n = x / s,  s = sqrt((|x|^2 + M_k)/64 + eps)  with M_k = 0.

Stage 2 moves
-------------
Her moves, per layer, statistics over all positions of the 512 'stats' windows,
computed from the CURRENT network when the move is applied:
  head_off, head_mean (training mean of the head output into b_O), head_mean_pos (per
  position, into a per-layer bias B_attn of T x d, live once used), attn_uniform,
  attn_fixed (mean pattern; costs T(T-1)/2), qk_rank / ov_rank (activation-weighted
  SVD, nnprune.transformer.weighted_lowrank, on the augmented inputs [x/s, 1/s, 1]
  when the head has biases), unit_sat (GELU unit off, b -> -inf), unit_mean, unit_merge
  (top n_merge correlated pairs per layer, chosen at refresh).
stream_mean (ours, through the exact LayerNorm route): rotate the stream so that a unit
  direction u is coordinate j; at each LayerNorm site k in depth order replace x_j by its
  training mean m_k, i.e. add m_k * (row j of every matrix reading site k) to a bias
  applied BEFORE the division by s (pQ, pK, pV, p1, pU), and add m_k^2 to M_k; then
  delete coordinate j from every writer.  Exact for the network in which u.x is
  replaced by m_k at every LayerNorm input (test_gpt_adapter.py).  Candidates are
  'fresh slots': the smallest eigen-directions of the pooled (n_pool) and per-site
  (n_each) centred covariances of the LayerNorm inputs, recomputed when applied.
Single weights ('zero', chunks of 64) on every matrix, as in her W+S arms.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "NNpruning"))
from nnprune.core import ACTIVE, REMOVED, Adapter, Move, State, zero_entries  # noqa: E402
from nnprune.transformer import weighted_lowrank  # noqa: E402
import gpt_toy as G  # noqa: E402

HER_KINDS = ("head_off", "head_mean", "head_mean_pos", "attn_uniform", "attn_fixed", "qk_rank", "ov_rank",
             "unit_sat", "unit_mean", "unit_merge")
STREAM_KINDS = ("stream_mean",)
# stream-indexed tensors -> axis of the stream index
WRITERS = {"W_E": 1, "W_pos": 1, "W_O": 3, "b_O": 1, "B_attn": 2, "W_2": 1, "b_2": 1}
READERS = {"W_Q": 2, "W_K": 2, "W_V": 2, "W_1": 2, "W_U": 0}
STREAM_AX = {**WRITERS, **READERS}
HEAD_T = ("W_Q", "W_K", "W_V", "W_O", "b_Q", "pQ", "pK", "pV")


def make_data(seed=0, n_train=2048, n_stats=512, n_valA=512):
    text, _ = G.load_text()
    T = G.GPTConfig().T
    Xtr, Ytr = G.windows(text["train"], T, n=n_train, seed=seed)
    Xva, Yva = G.windows(text["val"], T)
    perm = torch.randperm(len(Yva), generator=torch.Generator().manual_seed(1000 + seed))
    h = len(Yva) // 2
    return {"train": (Xtr, Ytr), "stats": (Xtr[:n_stats], Ytr[:n_stats]), "val": (Xva, Yva),
            "valA": (Xva[perm[:n_valA]], Yva[perm[:n_valA]]), "valB": (Xva[perm[h:]], Yva[perm[h:]]),
            "test": G.windows(text["test"], T)}


def _rot(t, ax, R):
    return torch.movedim(torch.movedim(t, ax, -1) @ R.to(t.dtype), -1, ax).contiguous()


def _along(mask, ax):
    """(d,) bool: is any entry live at each stream coordinate."""
    return torch.movedim(mask, ax, -1).reshape(-1, mask.shape[ax]).any(0)


def householder(u, j, d):
    """Symmetric orthogonal H with H e_j = u (so x -> x H makes u.x coordinate j)."""
    e = torch.zeros(d, dtype=torch.double); e[j] = 1
    w = u.double() - e
    if float(w.norm()) < 1e-12:
        return torch.eye(d, dtype=torch.double)
    w = w / w.norm()
    return torch.eye(d, dtype=torch.double) - 2 * torch.outer(w, w)


class GPTAdapter(Adapter):
    loss_type = "categorical"
    prunable = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_1", "W_2", "W_U"]
    site_layer = {n: i for i, n in enumerate(prunable)}       # only used by the 'layer' tie-breaks

    def __init__(self, data, cfg: G.GPTConfig, n_pool=6, n_each=1):
        super().__init__(data)
        self.cfg = cfg
        self.L, self.H, self.d, self.dh, self.dm, self.T, self.V = cfg.L, cfg.H, cfg.d, cfg.dh, cfg.d_mlp, cfg.T, cfg.V
        self.causal = torch.tril(torch.ones(cfg.T, cfg.T, dtype=torch.bool))
        self.n_pool, self.n_each = n_pool, n_each
        self.n_sites = 2 * cfg.L + 1
        self._cache_state, self._cache, self._cur = None, None, None

    # ---- loss: one sample = one window, loss = mean over its positions --------------------------
    def loss_from_logits(self, z, y):
        if z.dim() == 3:
            return F.cross_entropy(z.flatten(0, 1), y.flatten(), reduction="none").view(y.shape).mean(-1)
        return super().loss_from_logits(z, y)

    # ---- Stage 1 ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def stage1(self, P0: dict) -> State:
        P = {k: v.detach().double() for k, v in P0.items()}
        L, H, d, T = self.L, self.H, self.d, self.T
        C = torch.eye(d, dtype=torch.double) - 1.0 / d
        g1, be1, g2, be2 = P["ln1_g"], P["ln1_b"], P["ln2_g"], P["ln2_b"]
        bv = torch.einsum("ld,lhde->lhe", be1, P["W_V"]) + P["b_V"]
        Q = {
            "W_E": P["W_E"] @ C, "W_pos": P["W_pos"] @ C,
            "W_Q": P["W_Q"] * g1[:, None, :, None], "b_Q": torch.einsum("ld,lhde->lhe", be1, P["W_Q"]) + P["b_Q"],
            "W_K": P["W_K"] * g1[:, None, :, None], "W_V": P["W_V"] * g1[:, None, :, None],
            "W_O": P["W_O"] @ C, "b_O": (P["b_O"] + torch.einsum("lhe,lhed->ld", bv, P["W_O"])) @ C,
            "W_1": P["W_1"] * g2[:, None, :], "b_1": torch.einsum("lmd,ld->lm", P["W_1"], be2) + P["b_1"],
            "W_2": torch.einsum("de,lem->ldm", C, P["W_2"]), "b_2": P["b_2"] @ C,
            "W_U": P["lnf_g"][:, None] * P["W_U"], "b_U": P["lnf_b"] @ P["W_U"] + P["b_U"],
            "pQ": torch.zeros(L, H, self.dh, dtype=torch.double), "pK": torch.zeros(L, H, self.dh, dtype=torch.double),
            "pV": torch.zeros(L, H, self.dh, dtype=torch.double), "p1": torch.zeros(L, self.dm, dtype=torch.double),
            "pU": torch.zeros(self.V, dtype=torch.double), "M": torch.zeros(self.n_sites, dtype=torch.double),
            "B_attn": torch.zeros(L, T, d, dtype=torch.double),
        }
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in Q.items()}
        for k in ("pQ", "pK", "pV", "p1", "pU", "M", "B_attn"):
            masks[k][:] = False
        struct = {"head_alive": torch.ones(L, H, dtype=torch.bool), "head_fixed": torch.zeros(L, H, dtype=torch.bool),
                  "A_fixed": torch.zeros(L, H, T, T), "unit_type": torch.full((L, self.dm), ACTIVE, dtype=torch.int8),
                  "stream_alive": torch.ones(d, dtype=torch.bool)}
        s = State(Q, masks, struct)
        one = torch.full((d,), 1 / math.sqrt(d), dtype=torch.double)
        self._rotate(s, householder(one, d - 1, d))
        self._kill_coord(s, d - 1, STREAM_AX)
        s.struct["stream_alive"][d - 1] = False
        s.params = {k: v.float().contiguous() for k, v in s.params.items()}
        self.canonicalize(s)
        return s

    # ---- forward ---------------------------------------------------------------------------------------
    def _forward(self, P, S, X, cache=False, replace=None):
        """replace = (j, ms): at every LayerNorm input k, coordinate j is replaced by its
        mean over X (computed on the fly, appended to ms) before the LayerNorm reads it;
        the stream itself keeps x_j.  Used only to compute the means of a stream move."""
        n, T = X.shape
        causal = self.causal[:T, :T]
        x = P["W_E"][X] + P["W_pos"][:T]
        c = {"x": [], "s": [], "A": [], "ho": [], "act": []} if cache else None

        def read(x, k):
            if replace is not None:
                j, ms = replace
                ms.append(float(x[..., j].mean()))
                x = x.clone(); x[..., j] = ms[-1]
            return x, torch.sqrt((x.pow(2).sum(-1, keepdim=True) + P["M"][k]) / self.d + G.EPS)

        for l in range(self.L):
            xr, s1 = read(x, 2 * l)
            sh = s1[:, None]
            q = (torch.einsum("btd,hde->bhte", xr, P["W_Q"][l]) + P["pQ"][l][:, None]) / sh + P["b_Q"][l][:, None]
            k = (torch.einsum("btd,hde->bhte", xr, P["W_K"][l]) + P["pK"][l][:, None]) / sh
            v = (torch.einsum("btd,hde->bhte", xr, P["W_V"][l]) + P["pV"][l][:, None]) / sh
            alive_h = S["head_alive"][l].to(x.dtype)[:, None, None]
            if cache:
                A = torch.softmax((torch.einsum("bhte,bhse->bhts", q, k) / math.sqrt(self.dh))
                                  .masked_fill(~causal, float("-inf")), -1)
                A = torch.where(S["head_fixed"][l][:, None, None], S["A_fixed"][l][:, :T, :T].to(A.dtype), A)
                ho = torch.einsum("bhte,hed->bhtd", torch.einsum("bhts,bhse->bhte", A, v), P["W_O"][l]) * alive_h
                c["x"].append(x); c["s"].append(s1); c["A"].append(A); c["ho"].append(ho)
                attn_out = ho.sum(1)
            else:                                      # fused kernel; fixed patterns patched in
                z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                fx = S["head_fixed"][l]
                if bool(fx.any()):
                    z = z.clone()
                    z[:, fx] = torch.einsum("hts,bhse->bhte", S["A_fixed"][l][fx][:, :T, :T].to(v.dtype), v[:, fx])
                attn_out = torch.einsum("bhte,hed->btd", z * alive_h, P["W_O"][l])
            x = x + attn_out + P["b_O"][l] + P["B_attn"][l][:T]
            xr, s2 = read(x, 2 * l + 1)
            a = F.gelu((xr @ P["W_1"][l].T + P["p1"][l]) / s2 + P["b_1"][l])
            a = a * (S["unit_type"][l] == ACTIVE).to(a.dtype)
            if cache:
                c["x"].append(x); c["s"].append(s2); c["act"].append(a)
            x = x + a @ P["W_2"][l].T + P["b_2"][l]
        xr, sf = read(x, 2 * self.L)
        if cache:
            c["x"].append(x); c["s"].append(sf)
        return (xr @ P["W_U"] + P["pU"]) / sf + P["b_U"], c

    def logits(self, P, S, X):
        return torch.cat([self._forward(P, S, X[i:i + 1024])[0] for i in range(0, len(X), 1024)])

    def extra_k(self, s):
        return int(s.struct["head_fixed"].sum()) * (self.T * (self.T - 1) // 2)

    # ---- statistics of the current network (cached per state object) ----------------------------------
    @torch.no_grad()
    def _stats(self, s):
        if self._cache_state is not s:
            self._cache = self._forward(self.effective(s), s.struct, self.data["stats"][0], cache=True)[1]
            self._cache_state = s
        return self._cache

    def apply(self, state, move):
        self._cur = self._stats(state) if move.kind != "zero" else None
        return super().apply(state, move)

    # ---- bookkeeping helpers ---------------------------------------------------------------------------
    @staticmethod
    def _zero(s, name, index):
        mk = torch.zeros_like(s.masks[name]); mk[index] = True
        zero_entries(s, name, mk)

    def _kill_coord(self, s, j, names):
        for n in names:
            ax = STREAM_AX[n]
            if bool(s.masks[n].select(ax, j).any()):
                mk = torch.zeros_like(s.masks[n]); mk.select(ax, j).fill_(True)
                zero_entries(s, n, mk)

    @torch.no_grad()
    def _rotate(self, s, R):
        """x -> x R on the stream (R orthogonal, identity on dead coordinates)."""
        alive = s.struct["stream_alive"]
        P = self.effective(s)
        for n, ax in STREAM_AX.items():
            val = _rot(P[n], ax, R)
            m = s.masks[n]
            shape = [1] * m.dim(); shape[ax] = -1
            nm = (m.any(dim=ax, keepdim=True) & alive.view(shape)).contiguous()
            s.masks[n] = nm
            s.params[n] = (val * nm.to(val.dtype)).contiguous()

    def _kill_head(self, s, l, h):
        for n in HEAD_T:
            if bool(s.masks[n][l, h].any()):
                self._zero(s, n, (l, h))
        s.struct["head_alive"][l, h] = False
        s.struct["head_fixed"][l, h] = False

    def _zero_qk(self, s, l, h):
        for n in ("W_Q", "W_K", "b_Q", "pQ", "pK"):
            if bool(s.masks[n][l, h].any()):
                self._zero(s, n, (l, h))

    def _kill_unit(self, s, l, i):
        self._zero(s, "W_1", (l, i)); self._zero(s, "b_1", (l, i)); self._zero(s, "p1", (l, i))
        self._zero(s, "W_2", (l, slice(None), i))
        s.struct["unit_type"][l, i] = REMOVED

    def _fold_unit(self, s, l, i, c):
        W, M = s.params["W_2"], s.masks["W_2"]
        s.params["b_2"][l] += W[l, :, i] * M[l, :, i].to(W.dtype) * c
        self._kill_unit(s, l, i)

    def qk_dims(self, s, l, h):
        m = s.masks
        q = m["W_Q"][l, h].any(0) | m["pQ"][l, h] | m["b_Q"][l, h]
        k = m["W_K"][l, h].any(0) | m["pK"][l, h]
        return q & k

    def ov_dims(self, s, l, h):
        m = s.masks
        return (m["W_V"][l, h].any(0) | m["pV"][l, h]) & m["W_O"][l, h].any(1)

    # ---- canonicalisation (her rules + the stream rules) ------------------------------------------------
    def canonicalize(self, s):
        self._remask_dead(s)
        self._heads(s)
        self._units(s)
        st = s.struct
        alive = st["stream_alive"]
        read = torch.zeros(self.d, dtype=torch.bool)
        for n, ax in READERS.items():
            read |= _along(s.masks[n], ax)
        written = torch.zeros(self.d, dtype=torch.bool)
        for n, ax in WRITERS.items():
            written |= _along(s.masks[n], ax)
        for j in (alive & ~read).nonzero().flatten().tolist():
            self._kill_coord(s, j, WRITERS); alive[j] = False
        for j in (alive & ~written).nonzero().flatten().tolist():
            self._kill_coord(s, j, READERS); alive[j] = False
        self._heads(s)
        self._units(s)

    def _remask_dead(self, s):
        dead = ~s.struct["stream_alive"]
        if not bool(dead.any()):
            return
        for n, ax in STREAM_AX.items():
            if bool((_along(s.masks[n], ax) & dead).any()):
                shape = [1] * s.masks[n].dim(); shape[ax] = -1
                zero_entries(s, n, (dead.view(shape) & s.masks[n]))

    def _heads(self, s):
        st = s.struct
        for l in range(self.L):
            for h in range(self.H):
                if not bool(st["head_alive"][l, h]):
                    if any(bool(s.masks[n][l, h].any()) for n in HEAD_T):
                        self._kill_head(s, l, h)
                    continue
                qk, ov = self.qk_dims(s, l, h), self.ov_dims(s, l, h)
                for e in (~qk).nonzero().flatten().tolist():
                    for n in ("W_Q", "W_K"):
                        if bool(s.masks[n][l, h, :, e].any()):
                            self._zero(s, n, (l, h, slice(None), e))
                    for n in ("b_Q", "pQ", "pK"):
                        if bool(s.masks[n][l, h, e]):
                            self._zero(s, n, (l, h, e))
                for e in (~ov).nonzero().flatten().tolist():
                    if bool(s.masks["W_V"][l, h, :, e].any()):
                        self._zero(s, "W_V", (l, h, slice(None), e))
                    if bool(s.masks["pV"][l, h, e]):
                        self._zero(s, "pV", (l, h, e))
                    if bool(s.masks["W_O"][l, h, e].any()):
                        self._zero(s, "W_O", (l, h, e))
                if not bool(ov.any()):
                    self._kill_head(s, l, h)
                elif bool(st["head_fixed"][l, h]) and bool(qk.any()):
                    self._zero_qk(s, l, h)

    def _units(self, s):
        changed = True
        while changed:
            changed = False
            for l in range(self.L):
                t = s.struct["unit_type"][l]
                fan_out = s.masks["W_2"][l].any(0)
                fan_in = s.masks["W_1"][l].any(1) | s.masks["p1"][l]
                for i in ((t == ACTIVE) & ~fan_out).nonzero().flatten().tolist():
                    self._kill_unit(s, l, i); changed = True
                for i in ((s.struct["unit_type"][l] == ACTIVE) & ~fan_in).nonzero().flatten().tolist():
                    b = s.params["b_1"][l, i] if bool(s.masks["b_1"][l, i]) else torch.tensor(0.0)
                    self._fold_unit(s, l, i, float(F.gelu(b))); changed = True

    # ---- stream move ------------------------------------------------------------------------------------
    def stream_candidates(self, s, c):
        """Unit vectors (current basis, zero on dead coordinates); always n_slots of them
        (zero vectors pad the list)."""
        alive = s.struct["stream_alive"]
        Lc = alive.nonzero().flatten()
        n_slots = self.n_pool + self.n_each * self.n_sites
        if len(Lc) <= 1:
            return [torch.zeros(self.d, dtype=torch.double)] * n_slots
        Cs = []
        for x in c["x"]:
            a = x[..., Lc].reshape(-1, len(Lc)).double()
            a = a - a.mean(0)
            C = a.T @ a / len(a)
            Cs.append(C / max(float(torch.trace(C)), 1e-30))
        vecs = []
        for C, n in [(sum(Cs), self.n_pool)] + [(C, self.n_each) for C in Cs]:
            _, E = torch.linalg.eigh(C)
            for i in range(n):
                v = torch.zeros(self.d, dtype=torch.double)
                if i < E.shape[1]:
                    v[Lc] = E[:, i]
                vecs.append(v)
        return vecs

    @torch.no_grad()
    def _stream_mean(self, s, u):
        alive = s.struct["stream_alive"]
        if int(alive.sum()) <= 1:
            return False
        u = u.double().clone(); u[~alive] = 0
        if float(u.norm()) < 0.5:
            return False
        u = u / u.norm()
        j = int(alive.nonzero().max())
        self._rotate(s, householder(u, j, self.d))
        ms = []                                        # m_k, depth by depth, with earlier sites replaced
        self._forward(self.effective(s), s.struct, self.data["stats"][0], replace=(j, ms))
        for k, m in enumerate(ms):
            self._fold_site(s, k, j, m)
        self._kill_coord(s, j, WRITERS)
        alive[j] = False
        return True

    def _fold_site(self, s, k, j, m):
        P, Mk = s.params, s.masks
        if k < 2 * self.L:
            l = k // 2
            pairs = (("W_Q", "pQ"), ("W_K", "pK"), ("W_V", "pV")) if k % 2 == 0 else (("W_1", "p1"),)
            for W, p in pairs:
                if W == "W_1":
                    Wj, live = P[W][l, :, j] * Mk[W][l, :, j], Mk[W][l, :, j]
                    P[p][l] += m * Wj; Mk[p][l] |= live
                    if bool(live.any()):
                        self._zero(s, W, (l, slice(None), j))
                else:
                    Wj, live = P[W][l, :, j, :] * Mk[W][l, :, j, :], Mk[W][l, :, j, :]
                    P[p][l] += m * Wj; Mk[p][l] |= live
                    if bool(live.any()):
                        self._zero(s, W, (l, slice(None), j))
        else:
            Wj, live = P["W_U"][j] * Mk["W_U"][j], Mk["W_U"][j]
            P["pU"] += m * Wj; Mk["pU"] |= live
            if bool(live.any()):
                self._zero(s, "W_U", (j,))
        P["M"][k] += m * m
        Mk["M"][k] = True

    # ---- rank cuts on the augmented inputs --------------------------------------------------------------
    @torch.no_grad()
    def _rank_cut(self, s, c, kind, l, h):
        Lc = s.struct["stream_alive"].nonzero().flatten()
        nL = len(Lc)
        x, sc = c["x"][2 * l][..., Lc].double(), c["s"][2 * l].double()
        E = lambda n: s.params[n][l, h] * s.masks[n][l, h].to(s.params[n].dtype)
        if kind == "qk_rank":
            r = int(self.qk_dims(s, l, h).sum()) - 1
            if r < 1:
                return False
            inc_q = [n for n in ("pQ", "b_Q") if bool(s.masks[n][l, h].any())]
            inc_k = [n for n in ("pK",) if bool(s.masks[n][l, h].any())]
            feats = {"pQ": 1 / sc, "pK": 1 / sc, "b_Q": torch.ones_like(sc)}
            aq = torch.cat([x / sc] + [feats[n] for n in inc_q], -1).reshape(-1, nL + len(inc_q))
            ak = torch.cat([x / sc] + [feats[n] for n in inc_k], -1).reshape(-1, nL + len(inc_k))
            WQ = torch.cat([E("W_Q")[Lc]] + [E(n)[None] for n in inc_q]).double()
            WK = torch.cat([E("W_K")[Lc]] + [E(n)[None] for n in inc_k]).double()
            A, B = weighted_lowrank(WQ @ WK.T, aq.T @ aq / len(aq), ak.T @ ak / len(ak), r)
            for F_, W, inc in ((A, "W_Q", inc_q), (B, "W_K", inc_k)):
                self._write_factor(s, W, inc, l, h, Lc, F_, r)
            return True
        r = int(self.ov_dims(s, l, h).sum()) - 1
        if r < 1:
            return False
        inc_v = [n for n in ("pV",) if bool(s.masks[n][l, h].any())]
        av = torch.cat([x / sc] + [1 / sc for _ in inc_v], -1)                        # (n, T, nL + .)
        y = torch.einsum("nts,nsk->ntk", c["A"][l][:, h].double(), av).reshape(-1, av.shape[-1])
        WV = torch.cat([E("W_V")[Lc]] + [E(n)[None] for n in inc_v]).double()
        A, B = weighted_lowrank(WV @ E("W_O")[:, Lc].double(), y.T @ y / len(y),
                                torch.eye(nL, dtype=torch.double), r)
        self._write_factor(s, "W_V", inc_v, l, h, Lc, A, r)
        self._zero(s, "W_O", (l, h))
        rows = torch.arange(r)
        s.params["W_O"][l, h][rows[:, None], Lc[None, :]] = B.T
        s.masks["W_O"][l, h][rows[:, None], Lc[None, :]] = True
        return True

    def _write_factor(self, s, W, inc, l, h, Lc, F_, r):
        for n in [W] + inc:
            self._zero(s, n, (l, h))
        cols = torch.arange(r)
        s.params[W][l, h][Lc[:, None], cols[None, :]] = F_[:len(Lc)]
        s.masks[W][l, h][Lc[:, None], cols[None, :]] = True
        for i, n in enumerate(inc):
            s.params[n][l, h][:r] = F_[len(Lc) + i]
            s.masks[n][l, h][:r] = True

    # ---- refresh, candidates, moves -----------------------------------------------------------------------
    @torch.no_grad()
    def refresh_extra(self, s, cfg):
        c = self._stats(s)
        pairs = []
        for l in range(self.L):
            ids = (s.struct["unit_type"][l] == ACTIVE).nonzero().flatten()
            if ids.numel() < 2 or cfg.n_merge <= 0:
                continue
            A = c["act"][l][..., ids].reshape(-1, len(ids))
            Ac = A - A.mean(0)
            sd = Ac.pow(2).mean(0).sqrt()
            ok = sd > 1e-8
            Cr = (Ac.T @ Ac) / len(A) / (sd[:, None] * sd[None, :]).clamp_min(1e-16)
            Cr = Cr * (ok[:, None] & ok[None, :])
            iu = torch.triu_indices(len(ids), len(ids), 1)
            vals = Cr[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                a_, b_ = int(ids[iu[0, t_]]), int(ids[iu[1, t_]])
                pairs += [(l, a_, b_), (l, b_, a_)]
        return {"merge_pairs": pairs}

    def structural_moves(self, s, ctx, cfg):
        st, mv = s.struct, []
        for l in range(self.L):
            for h in range(self.H):
                if not bool(st["head_alive"][l, h]):
                    continue
                site = f"L{l}h{h}"
                for k in ("head_off", "head_mean", "head_mean_pos"):
                    mv.append(Move(k, site, l, {"l": l, "h": h}))
                qk = int(self.qk_dims(s, l, h).sum())
                if not bool(st["head_fixed"][l, h]) and qk > 0:
                    mv.append(Move("attn_uniform", site, l, {"l": l, "h": h}))
                    mv.append(Move("attn_fixed", site, l, {"l": l, "h": h}))
                    if qk > 1:
                        mv.append(Move("qk_rank", site, l, {"l": l, "h": h}))
                if int(self.ov_dims(s, l, h).sum()) > 1:
                    mv.append(Move("ov_rank", site, l, {"l": l, "h": h}))
            for i in (st["unit_type"][l] == ACTIVE).nonzero().flatten().tolist():
                mv.append(Move("unit_sat", f"mlp{l}", l, {"l": l, "i": i}))
                mv.append(Move("unit_mean", f"mlp{l}", l, {"l": l, "i": i}))
        for (l, i, i2) in ctx.get("merge_pairs", []):
            mv.append(Move("unit_merge", f"mlp{l}", l, {"l": l, "i": i, "i2": i2}))
        for slot in range(self.n_pool + self.n_each * self.n_sites):
            mv.append(Move("stream_mean", "stream", 0, {"slot": slot}))
        return mv

    @torch.no_grad()
    def apply_structural(self, s, m):
        p, st, c = m.payload, s.struct, self._cur
        if m.kind == "stream_mean":
            return self._stream_mean(s, self.stream_candidates(s, c)[p["slot"]])
        l = p["l"]
        if m.kind.startswith("unit_"):
            i = p["i"]
            if int(st["unit_type"][l, i]) != ACTIVE:
                return False
            if m.kind == "unit_sat":
                self._kill_unit(s, l, i)
                return True
            if m.kind == "unit_mean":
                self._fold_unit(s, l, i, float(c["act"][l][..., i].mean()))
                return True
            i2 = p["i2"]
            if int(st["unit_type"][l, i2]) != ACTIVE:
                return False
            a, b = c["act"][l][..., i].flatten().double(), c["act"][l][..., i2].flatten().double()
            bc = b - b.mean()
            alpha = float(((a - a.mean()) * bc).mean() / bc.pow(2).mean().clamp_min(1e-16))
            beta = float(a.mean() - alpha * b.mean())
            W, M = s.params["W_2"], s.masks["W_2"]
            col = W[l, :, i] * M[l, :, i].to(W.dtype)
            W[l, :, i2] = W[l, :, i2] * M[l, :, i2].to(W.dtype) + alpha * col
            M[l, :, i2] = M[l, :, i2] | M[l, :, i]
            s.params["b_2"][l] += beta * col
            self._kill_unit(s, l, i)
            return True
        h = p["h"]
        if not bool(st["head_alive"][l, h]):
            return False
        if m.kind == "head_off":
            self._kill_head(s, l, h)
            return True
        if m.kind == "head_mean":
            s.params["b_O"][l] += c["ho"][l][:, h].mean((0, 1))
            self._kill_head(s, l, h)
            return True
        if m.kind == "head_mean_pos":
            alive = st["stream_alive"]
            s.params["B_attn"][l] += c["ho"][l][:, h].mean(0) * alive.to(torch.float32)
            s.masks["B_attn"][l] |= alive[None, :].expand(self.T, -1)
            self._kill_head(s, l, h)
            return True
        if m.kind in ("attn_uniform", "attn_fixed", "qk_rank"):
            if bool(st["head_fixed"][l, h]) or int(self.qk_dims(s, l, h).sum()) == 0:
                return False
            if m.kind == "attn_uniform":
                self._zero_qk(s, l, h)
                return True
            if m.kind == "attn_fixed":
                st["A_fixed"][l, h] = c["A"][l][:, h].mean(0)
                st["head_fixed"][l, h] = True
                self._zero_qk(s, l, h)
                return True
            return self._rank_cut(s, c, "qk_rank", l, h)
        if m.kind == "ov_rank":
            return self._rank_cut(s, c, "ov_rank", l, h)
        raise ValueError(m.kind)


def load(L=2, seed=0, root=None, **kw):
    """-> (adapter, Stage-1 state, base dict) for results_gpt/L<L>_seed<seed>."""
    root = root or os.path.join(HERE, "results_gpt")
    base = torch.load(os.path.join(root, f"L{L}_seed{seed}", "base_state.pt"), weights_only=False)
    ad = GPTAdapter(make_data(seed), G.GPTConfig(**base["cfg"]), **kw)
    return ad, ad.stage1(base["params"]), base
