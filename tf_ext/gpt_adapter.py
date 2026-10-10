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
Stream moves at finer granularity (STREAM_FINE_KINDS).  A reader sees n = x / s; at site k:
  stream_read (one reader matrix R at site k): n_j -> c = E[n_j] for R only, folded into R's
      existing post-division bias (W_Q -> b_Q; W_K: dropped, a constant added to every key
      is ignored by the softmax; W_V -> b_O, attention rows sum to 1; W_1 -> b_1; W_U ->
      b_U).  Frees R's row j; no new numbers; sigma unchanged.
  stream_site (all readers at site k): the same for every reader at k, and inside sigma_k
      x_j^2 -> E[x_j^2] (struct site_on[k, j] = False, M_k += E[x_j^2]).  Site k then no
      longer depends on x_j at all.
  Writers die by depth (canonicalize): a writer's column j is deleted once no site after it
  reads x_j, through a matrix row or through sigma.
  Basis: these moves either act on an existing coordinate j, or rotate a direction u of the
  'pristine' coordinates (alive, never touched by a fine move) onto a coordinate first;
  rotations never mix touched coordinates, whose read pattern differs.  stream_mean
  likewise rotates only pristine coordinates.  Candidates: smallest-variance directions of
  n at site k (n_site per site); per reader, the one among the 6 smallest minimising
  Var(u.n) |W^T u|^2; and every live (site, coordinate) / (reader, coordinate) pair of
  touched coordinates.
  stream_write (one writer: token embedding, positions, attention output of layer l, MLP
      output of layer l): its contribution to x_j -> its training mean (over inputs and
      positions), folded into a bias (embedding -> W_pos column j; positions -> W_E column
      j; attention -> b_O[l, j]; MLP -> b_2[l, j]); the writer's column j is deleted.
  stream_mean on a touched coordinate j (payload j, no rotation): the global replacement
      at the sites that still read x_j, then all writers of j deleted.  Exact when every
      writer of j has been replaced by a constant (x_j is then constant at every site).
New moves (NEW_KINDS):
  emb_rank / pos_rank / unemb_rank  rank cut of W_E / W_pos / W_U by her activation-weighted
      SVD (weights: group frequencies of the characters for W_E, uniform positions for
      W_pos, the covariance of the final normalised stream x/s for W_U; identity on the
      output side).  Factors are counted as r (rows + cols), as her qk/ov factors are, so
      the first cut of a dense matrix jumps to the largest rank at which the factors are
      cheaper than the dense matrix; later cuts remove one rank.
  token_merge (the realisation of 'token_mean')  character c joins the group of c2: the
      group shares one embedding, the frequency-weighted mean of the two groups' vectors
      (the L2 projection, as in unit_merge); c's row is freed.  A plain 'replace c by a
      stored mean vector' frees nothing the first time (the mean costs a row), so her rule
      would never take it.  Candidates: the n_tok pairs of group representatives with the
      smallest  f1 f2 / (f1 + f2) |E1 - E2|^2.
  head_merge  heads h, h2 of one layer with similar patterns: h is deleted and h2, keeping
      its own pattern, gets the summed value-output map  N_h + N_h2  (on the augmented
      inputs), compressed to r = min(d_h, rank_h + rank_h2) by her ov_rank SVD weighted by
      the inputs averaged with h2's pattern (exact when the patterns are equal and the
      ranks fit in d_h).  All ordered pairs of live heads are candidates.
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
STREAM_FINE_KINDS = ("stream_site", "stream_read")
STREAM_WRITE_KINDS = ("stream_write",)
NEW_KINDS = ("emb_rank", "pos_rank", "unemb_rank", "token_merge", "head_merge")
# stream-indexed tensors -> axis of the stream index
WRITERS = {"W_E": 1, "W_pos": 1, "W_O": 3, "b_O": 1, "B_attn": 2, "W_2": 1, "b_2": 1, "E_b": 1, "Pos_b": 1}
READERS = {"W_Q": 2, "W_K": 2, "W_V": 2, "W_1": 2, "W_U": 0, "U_a": 0}
FACTORS = {"emb_rank": ("W_E", "E_a", "E_b"), "pos_rank": ("W_pos", "Pos_a", "Pos_b"),
           "unemb_rank": ("W_U", "U_a", "U_b")}
STREAM_AX = {**WRITERS, **READERS}
HEAD_T = ("W_Q", "W_K", "W_V", "W_O", "b_Q", "pQ", "pK", "pV")


def make_data(seed=0, n_train=2048, n_stats=512, n_valA=512):
    text, _ = G.load_text()
    T = G.GPTConfig().T
    Xtr, Ytr = G.windows(text["train"], T, n=n_train, seed=seed)
    Xva, Yva = G.windows(text["val"], T)
    freq = torch.bincount(text["train"], minlength=G.GPTConfig().V).double()
    perm = torch.randperm(len(Yva), generator=torch.Generator().manual_seed(1000 + seed))
    h = len(Yva) // 2
    return {"train": (Xtr, Ytr), "stats": (Xtr[:n_stats], Ytr[:n_stats]), "val": (Xva, Yva),
            "valA": (Xva[perm[:n_valA]], Yva[perm[:n_valA]]), "valB": (Xva[perm[h:]], Yva[perm[h:]]),
            "test": G.windows(text["test"], T), "freq": freq / freq.sum()}


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

    def __init__(self, data, cfg: G.GPTConfig, n_pool=6, n_each=1, n_tok=16, n_site=2):
        super().__init__(data)
        self.cfg = cfg
        self.L, self.H, self.d, self.dh, self.dm, self.T, self.V = cfg.L, cfg.H, cfg.d, cfg.dh, cfg.d_mlp, cfg.T, cfg.V
        self.causal = torch.tril(torch.ones(cfg.T, cfg.T, dtype=torch.bool))
        self.n_pool, self.n_each, self.n_tok, self.n_site = n_pool, n_each, n_tok, n_site
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
            "E_a": torch.zeros(self.V, d, dtype=torch.double), "E_b": torch.zeros(d, d, dtype=torch.double),
            "Pos_a": torch.zeros(T, T, dtype=torch.double), "Pos_b": torch.zeros(T, d, dtype=torch.double),
            "U_a": torch.zeros(d, d, dtype=torch.double), "U_b": torch.zeros(d, self.V, dtype=torch.double),
        }
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in Q.items()}
        for k in ("pQ", "pK", "pV", "p1", "pU", "M", "B_attn", "E_a", "E_b", "Pos_a", "Pos_b", "U_a", "U_b"):
            masks[k][:] = False
        struct = {"head_alive": torch.ones(L, H, dtype=torch.bool), "head_fixed": torch.zeros(L, H, dtype=torch.bool),
                  "A_fixed": torch.zeros(L, H, T, T), "unit_type": torch.full((L, self.dm), ACTIVE, dtype=torch.int8),
                  "stream_alive": torch.ones(d, dtype=torch.bool), "tok_group": torch.arange(self.V),
                  "site_on": torch.ones(self.n_sites, d, dtype=torch.bool), "touched": torch.zeros(d, dtype=torch.bool)}
        s = State(Q, masks, struct)
        one = torch.full((d,), 1 / math.sqrt(d), dtype=torch.double)
        self._rotate(s, householder(one, d - 1, d))
        self._kill_coord(s, d - 1, STREAM_AX)
        s.struct["stream_alive"][d - 1] = False
        s.params = {k: v.float().contiguous() for k, v in s.params.items()}
        self.canonicalize(s)
        return s

    # ---- forward ---------------------------------------------------------------------------------------
    def _forward(self, P, S, X, cache=False, replace=None, read_over=None, write_over=None):
        """replace = (j, ms): at every LayerNorm input k, coordinate j is replaced by its
        mean over X (computed on the fly, appended to ms) before the LayerNorm reads it;
        the stream itself keeps x_j.  Used only to compute the means of a stream move.
        read_over = {(k, reader): (j, c)}: reader at site k sees n_j = c (tests only).
        write_over = {writer: (j, mu)}: that writer's contribution to x_j is mu (tests only)."""
        n, T = X.shape
        causal = self.causal[:T, :T]
        wo = write_over or {}

        def wover(w, contrib):
            if w not in wo:
                return contrib
            j, mu = wo[w]
            contrib = contrib.clone(); contrib[..., j] = mu
            return contrib

        x = wover("emb", (P["W_E"] + P["E_a"] @ P["E_b"])[S["tok_group"][X]]) \
            + wover("pos", (P["W_pos"] + P["Pos_a"] @ P["Pos_b"])[:T].expand(n, T, -1))
        c = {"x": [], "s": [], "A": [], "ho": [], "act": []} if cache else None

        def read(x, k):
            if replace is not None:
                j, ms = replace
                ms.append(float(x[..., j].mean()))
                x = x.clone(); x[..., j] = ms[-1]
            on = S["site_on"][k].to(x.dtype)
            return x, torch.sqrt(((x.pow(2) * on).sum(-1, keepdim=True) + P["M"][k]) / self.d + G.EPS)

        def inp(xr, s_, k, name):
            if read_over is None or (k, name) not in read_over:
                return xr
            j, cval = read_over[(k, name)]
            xr = xr.clone(); xr[..., j] = cval * s_[..., 0]
            return xr

        for l in range(self.L):
            xr, s1 = read(x, 2 * l)
            sh = s1[:, None]
            q = (torch.einsum("btd,hde->bhte", inp(xr, s1, 2 * l, "W_Q"), P["W_Q"][l]) + P["pQ"][l][:, None]) / sh \
                + P["b_Q"][l][:, None]
            k = (torch.einsum("btd,hde->bhte", inp(xr, s1, 2 * l, "W_K"), P["W_K"][l]) + P["pK"][l][:, None]) / sh
            v = (torch.einsum("btd,hde->bhte", inp(xr, s1, 2 * l, "W_V"), P["W_V"][l]) + P["pV"][l][:, None]) / sh
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
            x = x + wover(("attn", l), attn_out + P["b_O"][l] + P["B_attn"][l][:T])
            xr, s2 = read(x, 2 * l + 1)
            a = F.gelu((inp(xr, s2, 2 * l + 1, "W_1") @ P["W_1"][l].T + P["p1"][l]) / s2 + P["b_1"][l])
            a = a * (S["unit_type"][l] == ACTIVE).to(a.dtype)
            if cache:
                c["x"].append(x); c["s"].append(s2); c["act"].append(a)
            x = x + wover(("mlp", l), a @ P["W_2"][l].T + P["b_2"][l])
        xr, sf = read(x, 2 * self.L)
        if cache:
            c["x"].append(x); c["s"].append(sf)
        return (inp(xr, sf, 2 * self.L, "W_U") @ (P["W_U"] + P["U_a"] @ P["U_b"]) + P["pU"]) / sf + P["b_U"], c

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

    def pristine(self, s):
        return s.struct["stream_alive"] & ~s.struct["touched"]

    @torch.no_grad()
    def _rotate(self, s, R, coords=None):
        """x -> x R on the stream.  R is orthogonal and acts only on the coordinates `coords`
        (default: all alive ones); masks of the other coordinates are kept as they are."""
        alive = s.struct["stream_alive"]
        coords = alive if coords is None else coords
        P = self.effective(s)
        for n, ax in STREAM_AX.items():
            val = _rot(P[n], ax, R)
            m = s.masks[n]
            shape = [1] * m.dim(); shape[ax] = -1
            nm = torch.where(coords.view(shape), m.any(dim=ax, keepdim=True) & alive.view(shape), m).contiguous()
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
        self._factors(s)
        self._heads(s)
        self._units(s)
        st = s.struct
        alive = st["stream_alive"]
        self._writer_death(s)
        written = torch.zeros(self.d, dtype=torch.bool)
        for n, ax in WRITERS.items():
            written |= _along(s.masks[n], ax)
        for j in (alive & ~written).nonzero().flatten().tolist():
            self._kill_coord(s, j, READERS); alive[j] = False
        self._heads(s)
        self._units(s)

    def site_readers(self, k):
        if k == 2 * self.L:
            return [("W_U", None)]
        l = k // 2
        return [("W_Q", l), ("W_K", l), ("W_V", l)] if k % 2 == 0 else [("W_1", l)]

    def _row_live(self, s, name, l):
        """(d,) bool: is row j of reader `name` (layer l) live."""
        m = s.masks
        if name == "W_U":
            return m["W_U"].any(1) | m["U_a"].any(1)
        if name == "W_1":
            return m["W_1"][l].any(0)
        return m[name][l].any(0).any(-1)

    def _reads_at(self, s, k):
        r = s.struct["site_on"][k].clone()
        for name, l in self.site_readers(k):
            r |= self._row_live(s, name, l)
        return r & s.struct["stream_alive"]

    def _writer_death(self, s):
        """Delete a writer's column j when no site after the writer reads x_j."""
        reads_after = [None] * (self.n_sites + 1)
        reads_after[self.n_sites] = torch.zeros(self.d, dtype=torch.bool)
        for k in range(self.n_sites - 1, -1, -1):
            reads_after[k] = reads_after[k + 1] | self._reads_at(s, k)
        groups = [(n, None, 0) for n in ("W_E", "W_pos", "E_b", "Pos_b")]
        for l in range(self.L):
            groups += [(n, l, 2 * l + 1) for n in ("W_O", "b_O", "B_attn")] + [(n, l, 2 * l + 2) for n in ("W_2", "b_2")]
        for n, l, k0 in groups:
            dead = ~reads_after[k0]
            m = s.masks[n] if l is None else s.masks[n][l]
            ax = WRITERS[n] - (0 if l is None else 1)
            if bool((_along(m, ax) & dead).any()):
                shape = [1] * m.dim(); shape[ax] = -1
                mk = torch.zeros_like(s.masks[n])
                if l is None:
                    mk |= dead.view(shape) & m
                else:
                    mk[l] = dead.view(shape) & m
                zero_entries(s, n, mk)

    def _factors(self, s):
        """A factor dimension is live only if both its column of a and its row of b are."""
        for a, b in (("E_a", "E_b"), ("Pos_a", "Pos_b"), ("U_a", "U_b")):
            live = s.masks[a].any(0) & s.masks[b].any(1)
            if bool((s.masks[a].any(0) & ~live).any()):
                zero_entries(s, a, s.masks[a] & ~live[None, :])
            if bool((s.masks[b].any(1) & ~live).any()):
                zero_entries(s, b, s.masks[b] & ~live[:, None])

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
        Lc = self.pristine(s).nonzero().flatten()
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
    def _stream_mean_coord(self, s, j):
        """The global replacement on an existing (touched) coordinate j, no rotation."""
        st = s.struct
        if not (bool(st["stream_alive"][j]) and bool(st["touched"][j])) or int(st["stream_alive"].sum()) <= 1:
            return False
        ms = []
        self._forward(self.effective(s), st, self.data["stats"][0], replace=(j, ms))
        for k, m in enumerate(ms):
            self._fold_site(s, k, j, m)
        self._kill_coord(s, j, WRITERS)
        st["stream_alive"][j] = False
        return True

    @torch.no_grad()
    def _stream_mean(self, s, u):
        alive, pr = s.struct["stream_alive"], self.pristine(s)
        if int(alive.sum()) <= 1 or int(pr.sum()) < 1:
            return False
        u = u.double().clone(); u[~pr] = 0
        if float(u.norm()) < 0.5:
            return False
        u = u / u.norm()
        j = int(pr.nonzero().max())
        self._rotate(s, householder(u, j, self.d), pr)
        ms = []                                        # m_k, depth by depth, with earlier sites replaced
        self._forward(self.effective(s), s.struct, self.data["stats"][0], replace=(j, ms))
        for k, m in enumerate(ms):
            self._fold_site(s, k, j, m)
        self._kill_coord(s, j, WRITERS)
        alive[j] = False
        return True

    # ---- stream moves at finer granularity -----------------------------------------------------------
    def _reader_matrix(self, s, name, l):
        """(d, out) effective matrix of a reader."""
        E = self.effective(s)
        if name == "W_U":
            return E["W_U"] + E["U_a"] @ E["U_b"]
        if name == "W_1":
            return E["W_1"][l].T
        return E[name][l].permute(1, 0, 2).reshape(self.d, -1)

    def fine_candidates(self, s, c):
        """{('site', k): [u, ...], ('read', k, name): [u]} over pristine coordinates."""
        Lc = self.pristine(s).nonzero().flatten()
        out = {}
        for k in range(self.n_sites):
            if len(Lc) < 1:
                out[("site", k)] = [None] * self.n_site
                for name, l in self.site_readers(k):
                    out[("read", k, name)] = [None]
                continue
            n = (c["x"][k][..., Lc] / c["s"][k]).reshape(-1, len(Lc)).double()
            n = n - n.mean(0)
            C = n.T @ n / len(n)
            w, E = torch.linalg.eigh(C)
            full = lambda e: torch.zeros(self.d, dtype=torch.double).index_put_((Lc,), e)
            out[("site", k)] = [full(E[:, i]) if i < E.shape[1] else None for i in range(self.n_site)]
            for name, l in self.site_readers(k):
                W = self._reader_matrix(s, name, l)[Lc].double()
                best, bu = None, None
                for i in range(min(6, E.shape[1])):
                    sc = float(w[i]) * float((W.T @ E[:, i]).pow(2).sum())
                    if best is None or sc < best:
                        best, bu = sc, full(E[:, i])
                out[("read", k, name)] = [bu]
        return out

    def _to_coordinate(self, s, u):
        """Rotate the pristine coordinates so that u becomes one of them; return its index."""
        pr = self.pristine(s)
        u = u.double().clone(); u[~pr] = 0
        if int(pr.sum()) < 1 or float(u.norm()) < 0.5:
            return None
        u = u / u.norm()
        j = int(pr.nonzero().max())
        self._rotate(s, householder(u, j, self.d), pr)
        return j

    def _fold_post(self, s, k, j, cval, name, l):
        """Reader `name` at site k: n_j -> cval, folded into its post-division bias; row j deleted."""
        P, Mk, E = s.params, s.masks, self.effective(s)
        if name == "W_Q":
            P["b_Q"][l] += cval * E["W_Q"][l][:, j, :]
        elif name == "W_V":
            P["b_O"][l] += cval * torch.einsum("he,hed->d", E["W_V"][l][:, j, :], E["W_O"][l])
        elif name == "W_1":
            P["b_1"][l] += cval * E["W_1"][l][:, j]
        elif name == "W_U":
            P["b_U"] += cval * (E["W_U"][j] + E["U_a"][j] @ E["U_b"])
        # W_K: a constant added to every key does not change the softmax: nothing to fold
        if name == "W_U":
            for n in ("W_U", "U_a"):
                if bool(Mk[n][j].any()):
                    self._zero(s, n, (j,))
        elif bool(Mk[name][l][..., j, :].any() if name != "W_1" else Mk[name][l][:, j].any()):
            self._zero(s, name, (l, slice(None), j))

    @torch.no_grad()
    def _stream_fine(self, s, c, kind, k, name=None, u=None, j=None):
        st = s.struct
        if u is not None:
            vec = u
        elif j is not None and bool(st["stream_alive"][j]) and bool(st["touched"][j]):
            vec = torch.zeros(self.d, dtype=torch.double); vec[j] = 1
        else:
            return False
        xk, sk = c["x"][k].double(), c["s"][k].double()
        proj = xk @ vec.to(xk.dtype)                                   # u.x at site k (pre-move basis)
        cval = float((proj / sk[..., 0]).mean())
        second = float(proj.pow(2).mean())
        if u is not None:
            j = self._to_coordinate(s, u)
            if j is None:
                return False
        if kind == "stream_read":
            if not bool(self._row_live(s, name, None if name == "W_U" else k // 2)[j]):
                return False
            self._fold_post(s, k, j, cval, name, None if name == "W_U" else k // 2)
        else:
            if not bool(st["site_on"][k, j]):
                return False
            for nm, l in self.site_readers(k):
                self._fold_post(s, k, j, cval, nm, l)
            st["site_on"][k, j] = False
            s.params["M"][k] += second
            s.masks["M"][k] = True
        st["touched"][j] = True
        return True

    def writers(self):
        return ["emb", "pos"] + [(t, l) for l in range(self.L) for t in ("attn", "mlp")]

    def _writer_contrib(self, s, c, w):
        """(n, T, d) contribution of writer w to the stream on the 'stats' windows."""
        E = self.effective(s)
        X = self.data["stats"][0]
        if w == "emb":
            return (E["W_E"] + E["E_a"] @ E["E_b"])[s.struct["tok_group"][X]]
        if w == "pos":
            return (E["W_pos"] + E["Pos_a"] @ E["Pos_b"])[: X.shape[1]].expand(X.shape[0], -1, -1)
        t, l = w
        if t == "attn":
            return c["ho"][l].sum(1) + E["b_O"][l] + E["B_attn"][l][: X.shape[1]]
        return c["act"][l] @ E["W_2"][l].T + E["b_2"][l]

    def _writer_live(self, s, w):
        m = s.masks
        if w == "emb":
            return m["W_E"].any(0) | m["E_b"].any(0)
        if w == "pos":
            return m["W_pos"].any(0) | m["Pos_b"].any(0)
        t, l = w
        if t == "attn":
            return m["W_O"][l].any(0).any(0) | m["B_attn"][l].any(0)
        return m["W_2"][l].any(1)

    def write_candidates(self, s, c):
        Lc = self.pristine(s).nonzero().flatten()
        out = {}
        for w in self.writers():
            if len(Lc) < 1:
                out[w] = [None] * self.n_site
                continue
            a = self._writer_contrib(s, c, w)[..., Lc].reshape(-1, len(Lc)).double()
            a = a - a.mean(0)
            _, E = torch.linalg.eigh(a.T @ a / len(a))
            out[w] = [torch.zeros(self.d, dtype=torch.double).index_put_((Lc,), E[:, i]) if i < E.shape[1] else None
                      for i in range(self.n_site)]
        return out

    @torch.no_grad()
    def _stream_write(self, s, c, w, u=None, j=None):
        st = s.struct
        if u is not None:
            vec = u
        elif j is not None and bool(st["stream_alive"][j]) and bool(st["touched"][j]):
            vec = torch.zeros(self.d, dtype=torch.double); vec[j] = 1
        else:
            return False
        mu = float((self._writer_contrib(s, c, w).double() @ vec).mean())
        if w == "pos" and not bool(s.masks["W_E"].any()):
            return False                                   # no dense W_E column to take the constant
        if u is not None:
            j = self._to_coordinate(s, u)
            if j is None:
                return False
        if not bool(self._writer_live(s, w)[j]):
            return False
        P, M = s.params, s.masks
        if w == "emb":
            P["W_pos"][:, j] = P["W_pos"][:, j] * M["W_pos"][:, j] + mu; M["W_pos"][:, j] = True
            for n in ("W_E", "E_b"):
                if bool(M[n][:, j].any()):
                    self._zero(s, n, (slice(None), j))
        elif w == "pos":
            reps = st["tok_group"] == torch.arange(self.V)
            P["W_E"][reps, j] = P["W_E"][reps, j] * M["W_E"][reps, j] + mu; M["W_E"][reps, j] = True
            for n in ("W_pos", "Pos_b"):
                if bool(M[n][:, j].any()):
                    self._zero(s, n, (slice(None), j))
        else:
            t, l = w
            if t == "attn":
                P["b_O"][l, j] = mu; M["b_O"][l, j] = True
                for n, idx in (("W_O", (l, slice(None), slice(None), j)), ("B_attn", (l, slice(None), j))):
                    if bool(M[n][idx].any()):
                        self._zero(s, n, idx)
            else:
                P["b_2"][l, j] = mu; M["b_2"][l, j] = True
                if bool(M["W_2"][l, j].any()):
                    self._zero(s, "W_2", (l, j))
        st["touched"][j] = True
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
            E = self.effective(s)
            Wj = E["W_U"][j] + E["U_a"][j] @ E["U_b"]
            live = Mk["W_U"][j] | (Mk["U_a"][j][:, None] & Mk["U_b"]).any(0)
            P["pU"] += m * Wj; Mk["pU"] |= live
            for n in ("W_U", "U_a"):
                if bool(Mk[n][j].any()):
                    self._zero(s, n, (j,))
        if bool(s.struct["site_on"][k, j]):
            P["M"][k] += m * m
            Mk["M"][k] = True
            s.struct["site_on"][k, j] = False

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

    # ---- new moves: single-matrix rank cuts, token groups, head merge ---------------------------------------
    def _matrix(self, s, kind):
        """(rows, cols, dense live count, factor rank or None) of W_E / W_pos / W_U."""
        dense, a, b = FACTORS[kind]
        Lc = s.struct["stream_alive"].nonzero().flatten()
        if kind == "emb_rank":
            rows, cols = (s.struct["tok_group"] == torch.arange(self.V)).nonzero().flatten(), Lc
        elif kind == "pos_rank":
            rows, cols = torch.arange(self.T), Lc
        else:
            rows, cols = Lc, torch.arange(self.V)
        r = int((s.masks[a].any(0) & s.masks[b].any(1)).sum())
        return rows, cols, int(s.masks[dense].sum()), (r if r > 0 else None)

    def _target_rank(self, s, kind):
        rows, cols, n_dense, r = self._matrix(s, kind)
        if r is not None:
            return r - 1 if r > 1 else None
        r_new = (n_dense - 1) // (len(rows) + len(cols))           # largest r with r (rows + cols) < n_dense
        return min(r_new, len(rows), len(cols)) if r_new >= 1 else None

    @torch.no_grad()
    def _matrix_rank_cut(self, s, c, kind):
        r = self._target_rank(s, kind)
        if r is None:
            return False
        rows, cols, _, _ = self._matrix(s, kind)
        dense, a, b = FACTORS[kind]
        E = self.effective(s)
        W = (E[dense] + E[a] @ E[b])[rows[:, None], cols[None, :]].double()
        if kind == "emb_rank":
            g = s.struct["tok_group"]
            fr = torch.stack([self.data["freq"][g == int(i)].sum() for i in rows])
            Cl = torch.diag(fr / fr.sum())
        elif kind == "pos_rank":
            Cl = torch.eye(len(rows), dtype=torch.double)
        else:
            y = (c["x"][-1][..., rows] / c["s"][-1]).reshape(-1, len(rows)).double()
            Cl = y.T @ y / len(y)
        A, B = weighted_lowrank(W, Cl, torch.eye(len(cols), dtype=torch.double), r)
        for n in (dense, a, b):
            if bool(s.masks[n].any()):
                zero_entries(s, n, s.masks[n].clone())
        k = torch.arange(r)
        s.params[a][rows[:, None], k[None, :]] = A; s.masks[a][rows[:, None], k[None, :]] = True
        s.params[b][k[:, None], cols[None, :]] = B.T; s.masks[b][k[:, None], cols[None, :]] = True
        return True

    def _group_freq(self, s, c):
        return float(self.data["freq"][s.struct["tok_group"] == c].sum())

    @torch.no_grad()
    def _token_merge(self, s, c, c2):
        g = s.struct["tok_group"]
        if int(g[c]) != c or int(g[c2]) != c2 or c == c2:
            return False
        f1, f2 = self._group_freq(s, c), self._group_freq(s, c2)
        for n in ("W_E", "E_a"):
            P, M = s.params[n], s.masks[n]
            r1, r2 = P[c] * M[c].to(P.dtype), P[c2] * M[c2].to(P.dtype)
            P[c2] = (f1 * r1 + f2 * r2) / max(f1 + f2, 1e-30)
            M[c2] = M[c] | M[c2]
            if bool(M[c].any()):
                self._zero(s, n, (c,))
        g[g == c] = c2
        return True

    @torch.no_grad()
    def _head_merge(self, s, c, l, h, h2):
        st = s.struct
        if h == h2 or not (bool(st["head_alive"][l, h]) and bool(st["head_alive"][l, h2])):
            return False
        Lc = st["stream_alive"].nonzero().flatten()
        nL = len(Lc)
        r = min(self.dh, int(self.ov_dims(s, l, h).sum()) + int(self.ov_dims(s, l, h2).sum()))
        inc = bool(s.masks["pV"][l, h].any() or s.masks["pV"][l, h2].any())
        x, sc = c["x"][2 * l][..., Lc].double(), c["s"][2 * l].double()
        av = torch.cat([x / sc] + ([1 / sc] if inc else []), -1)
        y = torch.einsum("nts,nsk->ntk", c["A"][l][:, h2].double(), av).reshape(-1, av.shape[-1])
        E = lambda n, hh: (s.params[n][l, hh] * s.masks[n][l, hh].to(s.params[n].dtype)).double()
        N = 0
        for hh in (h, h2):
            WV = torch.cat([E("W_V", hh)[Lc]] + ([E("pV", hh)[None]] if inc else []))
            N = N + WV @ E("W_O", hh)[:, Lc]
        A, B = weighted_lowrank(N, y.T @ y / len(y), torch.eye(nL, dtype=torch.double), r)
        self._write_factor(s, "W_V", ["pV"] if inc else [], l, h2, Lc, A, r)
        self._zero(s, "W_O", (l, h2))
        k = torch.arange(r)
        s.params["W_O"][l, h2][k[:, None], Lc[None, :]] = B.T
        s.masks["W_O"][l, h2][k[:, None], Lc[None, :]] = True
        self._kill_head(s, l, h)
        return True

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
        g = s.struct["tok_group"]
        reps = (g == torch.arange(self.V)).nonzero().flatten()
        E = self.effective(s)
        Em = (E["W_E"] + E["E_a"] @ E["E_b"])[reps].double()
        f = torch.tensor([self._group_freq(s, int(i)) for i in reps], dtype=torch.double)
        cost = (f[:, None] * f[None, :] / (f[:, None] + f[None, :]).clamp_min(1e-30)) * torch.cdist(Em, Em).pow(2)
        iu = torch.triu_indices(len(reps), len(reps), 1)
        tok = []
        for t_ in torch.argsort(cost[iu[0], iu[1]])[: self.n_tok].tolist():
            a_, b_ = int(reps[iu[0, t_]]), int(reps[iu[1, t_]])
            tok.append((a_, b_) if f[iu[0, t_]] <= f[iu[1, t_]] else (b_, a_))     # rarer group joins the other
        return {"merge_pairs": pairs, "token_pairs": tok}

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
        alive_t = (st["stream_alive"] & st["touched"]).nonzero().flatten().tolist()
        for wi, w in enumerate(self.writers()):
            for i in range(self.n_site):
                mv.append(Move("stream_write", f"write{wi}", 0, {"writer": wi, "slot": i}))
            live = self._writer_live(s, w)
            for j in alive_t:
                if bool(live[j]):
                    mv.append(Move("stream_write", f"write{wi}", 0, {"writer": wi, "j": j}))
        for j in alive_t:
            mv.append(Move("stream_mean", "stream", 0, {"j": j}))
        for k in range(self.n_sites):
            for i in range(self.n_site):
                mv.append(Move("stream_site", f"site{k}", 0, {"site": k, "slot": i}))
            for name, l in self.site_readers(k):
                mv.append(Move("stream_read", f"site{k}.{name}", 0, {"site": k, "reader": name, "slot": 0}))
            for j in alive_t:
                if bool(st["site_on"][k, j]):
                    mv.append(Move("stream_site", f"site{k}", 0, {"site": k, "j": j}))
                for name, l in self.site_readers(k):
                    if bool(self._row_live(s, name, l)[j]):
                        mv.append(Move("stream_read", f"site{k}.{name}", 0, {"site": k, "reader": name, "j": j}))
        for kind in FACTORS:
            if self._target_rank(s, kind) is not None:
                mv.append(Move(kind, kind.split("_")[0], 0, {}))
        for (c1, c2) in ctx.get("token_pairs", []):
            mv.append(Move("token_merge", "emb", 0, {"c": c1, "c2": c2}))
        for l in range(self.L):
            live = st["head_alive"][l].nonzero().flatten().tolist()
            for h in live:
                for h2 in live:
                    if h != h2:
                        mv.append(Move("head_merge", f"L{l}h{h}", l, {"l": l, "h": h, "h2": h2}))
        return mv

    @torch.no_grad()
    def apply_structural(self, s, m):
        p, st, c = m.payload, s.struct, self._cur
        if m.kind == "stream_mean":
            if "j" in p:
                return self._stream_mean_coord(s, p["j"])
            return self._stream_mean(s, self.stream_candidates(s, c)[p["slot"]])
        if m.kind == "stream_write":
            w = self.writers()[p["writer"]]
            if "slot" in p:
                u = self.write_candidates(s, c)[w][p["slot"]]
                return False if u is None else self._stream_write(s, c, w, u=u)
            return self._stream_write(s, c, w, j=p["j"])
        if m.kind in STREAM_FINE_KINDS:
            k, name = p["site"], p.get("reader")
            if "slot" in p:
                key = ("site", k) if m.kind == "stream_site" else ("read", k, name)
                u = self.fine_candidates(s, c)[key][p["slot"]]
                return False if u is None else self._stream_fine(s, c, m.kind, k, name, u=u)
            return self._stream_fine(s, c, m.kind, k, name, j=p["j"])
        if m.kind in FACTORS:
            return self._matrix_rank_cut(s, c, m.kind)
        if m.kind == "token_merge":
            return self._token_merge(s, p["c"], p["c2"])
        if m.kind == "head_merge":
            return self._head_merge(s, c, p["l"], p["h"], p["h2"])
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
