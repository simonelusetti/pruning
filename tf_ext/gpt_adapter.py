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
    replaced by m_k at every LayerNorm input.  Candidates are
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
Key / value sharing (KV_KINDS; grouped-query attention as a pruning move).  struct k_group[l, h]
  (v_group) is the slot whose keys (values) head h uses; initially every head uses its own.
  A slot's W_K, pK (W_V, pV) belong to its group, not to a head: a key dimension survives while
  some member's query uses it, a value dimension while some member's W_O row uses it.
  k_share(h -> g): h (alone in its key group) drops its keys and uses slot g's; its queries are
      refitted, M_h = [W_Q; ...][W_K; ...]^T projected onto slot g's live key dimensions with her
      key weighting:  Q' = M_h C_k K_g (K_g^T C_k K_g)^{-1}  (exact if K_h = K_g G).
  v_share(h -> g): the same for values: O' = (V_g^T C_y V_g)^{-1} V_g^T C_y N_h, N_h = [W_V; pV] W_O,
      C_y from h's own pattern (exact if V_h = V_g G).
  qk_rank / ov_rank act on a whole group: the members' query factors (output maps) are stacked and
  cut together (block-diagonal / pooled weighting); for a group of one this is her move.
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
KV_KINDS = ("k_share", "v_share")
CLASSIC_MOVES = HER_KINDS
OUTDATED_MOVES = STREAM_KINDS + STREAM_FINE_KINDS + STREAM_WRITE_KINDS + NEW_KINDS + KV_KINDS
EXTENSION_MOVES = ()
MOVE_GROUPS = {
    "classic": CLASSIC_MOVES,
    "outdated": OUTDATED_MOVES,
    "extension": EXTENSION_MOVES,
    "fisher": (),
}
# stream-indexed tensors -> axis of the stream index
WRITERS = {"W_E": 1, "W_pos": 1, "W_O": 3, "b_O": 1, "B_attn": 2, "W_2": 1, "b_2": 1, "E_b": 1, "Pos_b": 1}
READERS = {"W_Q": 2, "W_K": 2, "W_V": 2, "W_1": 2, "W_U": 0, "U_a": 0}
FACTORS = {"emb_rank": ("W_E", "E_a", "E_b"), "pos_rank": ("W_pos", "Pos_a", "Pos_b"),
           "unemb_rank": ("W_U", "U_a", "U_b")}
STREAM_AX = {**WRITERS, **READERS}
HEAD_T = ("W_Q", "W_K", "W_V", "W_O", "b_Q", "pQ", "pK", "pV")
HEAD_OWN = ("W_Q", "b_Q", "pQ", "W_O")          # owned by the head; W_K, pK / W_V, pV are owned by its key / value group


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
    e = torch.zeros(d, dtype=torch.double, device=u.device); e[j] = 1
    w = u.double() - e
    if float(w.norm()) < 1e-12:
        return torch.eye(d, dtype=torch.double, device=u.device)
    w = w / w.norm()
    return torch.eye(d, dtype=torch.double, device=u.device) - 2 * torch.outer(w, w)


class GPTAdapter(Adapter):
    move_groups = MOVE_GROUPS
    loss_type = "categorical"
    fdist_sequence = True
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
                  "site_on": torch.ones(self.n_sites, d, dtype=torch.bool), "touched": torch.zeros(d, dtype=torch.bool),
                  "k_group": torch.arange(H).repeat(L, 1), "v_group": torch.arange(H).repeat(L, 1)}
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
            kg, vg = S["k_group"][l], S["v_group"][l]
            k = (torch.einsum("btd,hde->bhte", inp(xr, s1, 2 * l, "W_K"), P["W_K"][l][kg]) + P["pK"][l][kg][:, None]) / sh
            v = (torch.einsum("btd,hde->bhte", inp(xr, s1, 2 * l, "W_V"), P["W_V"][l][vg]) + P["pV"][l][vg][:, None]) / sh
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

    @torch.no_grad()
    def input_norms(self, s):
        """Input-column norms used by the repository's one-shot Wanda baseline.

        The returned tensors broadcast to the corresponding weight tensors.  As in
        the other adapters, ranking is global; this supplies Wanda's activation-aware
        score but does not change the core baseline to Wanda's per-output-row budget.
        """
        X = self.data["train"][0]
        P, S = self.effective(s), s.struct
        _, c = self._forward(P, S, X, cache=True)

        token_n = torch.bincount(X.reshape(-1), minlength=self.V).to(P["W_E"].dtype).sqrt()
        qkv_n, wo_n, w1_n, w2_n = [], [], [], []
        for l in range(self.L):
            n1 = c["x"][2 * l] / c["s"][2 * l]
            qkv_n.append(n1.pow(2).sum((0, 1)).sqrt())

            vg = S["v_group"][l]
            v = torch.einsum("btd,hde->bhte", n1, P["W_V"][l][vg])
            v = v + P["pV"][l][vg][None, :, None] / c["s"][2 * l][:, None]
            attended = torch.einsum("bhts,bhse->bhte", c["A"][l], v)
            wo_n.append(attended.pow(2).sum((0, 2)).sqrt())

            n2 = c["x"][2 * l + 1] / c["s"][2 * l + 1]
            w1_n.append(n2.pow(2).sum((0, 1)).sqrt())
            w2_n.append(c["act"][l].pow(2).sum((0, 1)).sqrt())

        nf = c["x"][-1] / c["s"][-1]
        return {
            "W_E": token_n[:, None],
            "W_pos": torch.full((self.T, 1), math.sqrt(len(X)), dtype=P["W_pos"].dtype,
                                device=X.device),
            "W_Q": torch.stack(qkv_n)[:, None, :, None],
            "W_K": torch.stack(qkv_n)[:, None, :, None],
            "W_V": torch.stack(qkv_n)[:, None, :, None],
            "W_O": torch.stack(wo_n)[:, :, :, None],
            "W_1": torch.stack(w1_n)[:, None, :],
            "W_2": torch.stack(w2_n)[:, None, :],
            "W_U": nf.pow(2).sum((0, 1)).sqrt()[:, None],
        }

    @torch.no_grad()
    def classical_unit_order(self, s):
        """One-shot mean-ablation order for all live MLP units and heads."""
        P = self.effective(s)
        _, c = self._forward(P, s.struct, self.data["train"][0], cache=True)
        items = []
        for l in range(self.L):
            act = c["act"][l].reshape(-1, self.dm)
            unit_score = act.var(0, unbiased=False) * P["W_2"][l].pow(2).sum(0)
            for i in (s.struct["unit_type"][l] == ACTIVE).nonzero().flatten().tolist():
                items.append((float(unit_score[i]), ("unit", l, i)))

            ho = c["ho"][l].permute(1, 0, 2, 3).reshape(self.H, -1, self.d)
            head_score = ho.var(1, unbiased=False).sum(1)
            for h in s.struct["head_alive"][l].nonzero().flatten().tolist():
                items.append((float(head_score[h]), ("head", l, h)))
        return [item for _, item in sorted(items, key=lambda z: z[0])]

    @torch.no_grad()
    def mean_move(self, s, item):
        """Construct the current-network mean replacement used by unit_baseline_path."""
        kind, l, i = item
        _, c = self._forward(self.effective(s), s.struct, self.data["train"][0], cache=True)
        if kind == "unit":
            mean = float(c["act"][l][..., i].mean())
            return Move("unit_mean", f"mlp{l}", l, {"l": l, "i": i, "mean": mean})
        mean = c["ho"][l][:, i].mean((0, 1))
        return Move("head_mean", f"L{l}h{i}", l, {"l": l, "h": i, "mean": mean})

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
        for n in HEAD_OWN:
            if bool(s.masks[n][l, h].any()):
                self._zero(s, n, (l, h))
        s.struct["head_alive"][l, h] = False
        s.struct["head_fixed"][l, h] = False

    def _zero_qk(self, s, l, h):
        for n in ("W_Q", "b_Q", "pQ"):                  # the keys belong to the group: cleaned up if unused
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

    def _q_side(self, s, l, h):
        m = s.masks
        return m["W_Q"][l, h].any(0) | m["pQ"][l, h] | m["b_Q"][l, h]

    def _k_slot(self, s, l, g):
        return s.masks["W_K"][l, g].any(0) | s.masks["pK"][l, g]

    def _v_slot(self, s, l, g):
        return s.masks["W_V"][l, g].any(0) | s.masks["pV"][l, g]

    def qk_dims(self, s, l, h):
        return self._q_side(s, l, h) & self._k_slot(s, l, int(s.struct["k_group"][l, h]))

    def ov_dims(self, s, l, h):
        return self._v_slot(s, l, int(s.struct["v_group"][l, h])) & s.masks["W_O"][l, h].any(1)

    def k_members(self, s, l, g):
        st = s.struct
        return [h for h in range(self.H) if bool(st["head_alive"][l, h]) and not bool(st["head_fixed"][l, h])
                and int(st["k_group"][l, h]) == g]

    def v_members(self, s, l, g):
        st = s.struct
        return [h for h in range(self.H) if bool(st["head_alive"][l, h]) and int(st["v_group"][l, h]) == g]

    # ---- ownership graph and canonicalisation ---------------------------------------------------------------
    #
    # Every live parameter entry belongs to exactly one node.  Nodes come in families; each family
    # states, for all of its nodes at once, when a node is unused (no live consumer) or unfed (no
    # live input), and what happens then: its entries are deleted, or (unit without fan-in) its
    # constant output is folded into the consumer's bias first.  canonicalize() applies every family
    # until none changes anything, so cascades across node types need no hand-written order.
    # A move only edits its own entries; a move that introduces a new kind of shared object adds a
    # family here.  is_canonical() is the invariant: no family would change the state.
    #
    #   family            node                    consumer / input                         rule
    #   dead_stream       dead stream coordinate  -                                        delete every entry on it
    #   factors           factor dimension k      a[:, k] and b[k, :] both live            else delete both sides
    #   qk_dims           query dims of a head    its key group has that key dimension     else delete the query side
    #                     key dims of a group     some non-fixed member's query uses it    else delete the key side
    #   ov_dims           W_O rows of a head      its value group has that value dimension else delete the row
    #                     value dims of a group   some member's W_O row uses it            else delete the value side
    #   heads             head                    some OV dimension live; fixed head: no QK  else delete the head
    #   units             MLP unit                fan-out live / fan-in live               delete / fold GELU(b) into b_2
    #   writer_columns    writer column j         a site after the writer reads x_j        else delete the column
    #   unwritten         stream coordinate       some writer entry live                   else x_j = 0: delete readers

    def _families(self):
        return (self._f_dead_stream, self._f_factors, self._f_qk_dims, self._f_ov_dims, self._f_heads,
                self._f_units, self._f_writer_columns, self._f_unwritten)

    def canonicalize(self, s):
        for _ in range(100):
            changed = False
            for fam in self._families():
                changed |= bool(fam(s))
            if not changed:
                return
        raise RuntimeError("canonicalisation did not reach a fixed point")

    def is_canonical(self, s):
        """True if no family would change s (checked on a copy)."""
        t = s.clone()
        return not any(bool(fam(t)) for fam in self._families())

    @staticmethod
    def _kill(s, name, mask):
        """Delete the live entries of `name` selected by the bool `mask`; return whether any were live."""
        sel = mask & s.masks[name]
        if bool(sel.any()):
            zero_entries(s, name, sel)
            return True
        return False

    def _f_dead_stream(self, s):
        dead = ~s.struct["stream_alive"]
        changed = False
        if bool(dead.any()):
            for n, ax in STREAM_AX.items():
                shape = [1] * s.masks[n].dim(); shape[ax] = -1
                changed |= self._kill(s, n, dead.view(shape).expand_as(s.masks[n]))
        return changed

    def _f_factors(self, s):
        changed = False
        for a, b in (("E_a", "E_b"), ("Pos_a", "Pos_b"), ("U_a", "U_b")):
            live = s.masks[a].any(0) & s.masks[b].any(1)
            changed |= self._kill(s, a, (~live)[None, :].expand_as(s.masks[a]))
            changed |= self._kill(s, b, (~live)[:, None].expand_as(s.masks[b]))
        return changed

    def _kill_dims(self, s, specs, l, h, dead):
        """Delete head dimensions `dead` of the listed tensors at (l, h).  spec: (name, matrix?)."""
        changed = False
        for n, matrix in specs:
            mk = torch.zeros_like(s.masks[n])
            if matrix == "cols":
                mk[l, h] = dead[None, :]
            elif matrix == "rows":
                mk[l, h] = dead[:, None]
            else:
                mk[l, h] = dead
            changed |= self._kill(s, n, mk)
        return changed

    def _f_qk_dims(self, s):
        """Query side of a head lives where its key group has a key; a key dimension of a group
        lives where some (non-fixed) member's query does."""
        changed = False
        qside = (("W_Q", "cols"), ("b_Q", None), ("pQ", None))
        for l in range(self.L):
            for h in range(self.H):
                if not bool(s.struct["head_alive"][l, h]):
                    continue
                dead = ~self.qk_dims(s, l, h)
                if bool(s.struct["head_fixed"][l, h]):
                    dead = torch.ones_like(dead)                # a fixed pattern has no QK consumer
                if bool(dead.any()):
                    changed |= self._kill_dims(s, qside, l, h, dead)
            for g in range(self.H):
                need = torch.zeros(self.dh, dtype=torch.bool, device=s.params["W_K"].device)
                for h in self.k_members(s, l, g):
                    need |= self._q_side(s, l, h)
                changed |= self._kill_dims(s, (("W_K", "cols"), ("pK", None)), l, g, ~need)
        return changed

    def _f_ov_dims(self, s):
        """W_O rows of a head live where its value group has a value; a value dimension of a group
        lives where some member's W_O row does."""
        changed = False
        for l in range(self.L):
            for h in range(self.H):
                if not bool(s.struct["head_alive"][l, h]):
                    continue
                dead = ~self.ov_dims(s, l, h)
                if bool(dead.any()):
                    changed |= self._kill_dims(s, (("W_O", "rows"),), l, h, dead)
            for g in range(self.H):
                need = torch.zeros(self.dh, dtype=torch.bool, device=s.params["W_V"].device)
                for h in self.v_members(s, l, g):
                    need |= s.masks["W_O"][l, h].any(1)
                changed |= self._kill_dims(s, (("W_V", "cols"), ("pV", None)), l, g, ~need)
        return changed

    def _f_heads(self, s):
        changed = False
        st = s.struct
        for l in range(self.L):
            for h in range(self.H):
                if bool(st["head_alive"][l, h]) and bool(self.ov_dims(s, l, h).any()):
                    continue
                if bool(st["head_alive"][l, h]) or any(bool(s.masks[n][l, h].any()) for n in HEAD_OWN):
                    self._kill_head(s, l, h)
                    changed = True
        return changed

    def _f_units(self, s):
        changed = False
        for l in range(self.L):
            t = s.struct["unit_type"][l]
            fan_out = s.masks["W_2"][l].any(0)
            for i in ((t == ACTIVE) & ~fan_out).nonzero().flatten().tolist():
                self._kill_unit(s, l, i); changed = True
            fan_in = s.masks["W_1"][l].any(1) | s.masks["p1"][l]
            for i in ((s.struct["unit_type"][l] == ACTIVE) & ~fan_in).nonzero().flatten().tolist():
                b = s.params["b_1"][l, i] if bool(s.masks["b_1"][l, i]) else s.params["b_1"].new_zeros(())
                self._fold_unit(s, l, i, float(F.gelu(b))); changed = True
        return changed

    def _f_writer_columns(self, s):
        reads_after = [None] * (self.n_sites + 1)
        reads_after[self.n_sites] = torch.zeros(self.d, dtype=torch.bool,
                                                device=s.struct["stream_alive"].device)
        for k in range(self.n_sites - 1, -1, -1):
            reads_after[k] = reads_after[k + 1] | self._reads_at(s, k)
        groups = [(n, None, 0) for n in ("W_E", "W_pos", "E_b", "Pos_b")]
        for l in range(self.L):
            groups += [(n, l, 2 * l + 1) for n in ("W_O", "b_O", "B_attn")] + [(n, l, 2 * l + 2) for n in ("W_2", "b_2")]
        changed = False
        for n, l, k0 in groups:
            dead = ~reads_after[k0]
            ax = WRITERS[n]
            shape = [1] * s.masks[n].dim(); shape[ax] = -1
            mk = dead.view(shape).expand_as(s.masks[n]).clone()
            if l is not None:
                keep = torch.zeros_like(mk); keep[l] = True
                mk &= keep
            changed |= self._kill(s, n, mk)
        return changed

    def _f_unwritten(self, s):
        alive = s.struct["stream_alive"]
        written = torch.zeros(self.d, dtype=torch.bool, device=alive.device)
        for n, ax in WRITERS.items():
            written |= _along(s.masks[n], ax)
        changed = False
        for j in (alive & ~written).nonzero().flatten().tolist():
            self._kill_coord(s, j, READERS); alive[j] = False
            changed = True
        return changed

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

    # ---- stream move ------------------------------------------------------------------------------------
    def stream_candidates(self, s, c):
        """Unit vectors (current basis, zero on dead coordinates); always n_slots of them
        (zero vectors pad the list)."""
        Lc = self.pristine(s).nonzero().flatten()
        n_slots = self.n_pool + self.n_each * self.n_sites
        if len(Lc) <= 1:
            return [torch.zeros(self.d, dtype=torch.double, device=Lc.device)] * n_slots
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
                v = torch.zeros(self.d, dtype=torch.double, device=Lc.device)
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
            full = lambda e: torch.zeros(self.d, dtype=torch.double, device=Lc.device).index_put_((Lc,), e)
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
            vg = s.struct["v_group"][l]
            P["b_O"][l] += cval * torch.einsum("he,hed->d", E["W_V"][l][vg][:, j, :], E["W_O"][l])
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
            vec = torch.zeros(self.d, dtype=torch.double, device=st["stream_alive"].device); vec[j] = 1
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
            out[w] = [torch.zeros(self.d, dtype=torch.double, device=Lc.device).index_put_((Lc,), E[:, i]) if i < E.shape[1] else None
                      for i in range(self.n_site)]
        return out

    @torch.no_grad()
    def _stream_write(self, s, c, w, u=None, j=None):
        st = s.struct
        if u is not None:
            vec = u
        elif j is not None and bool(st["stream_alive"][j]) and bool(st["touched"][j]):
            vec = torch.zeros(self.d, dtype=torch.double, device=st["stream_alive"].device); vec[j] = 1
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
            reps = st["tok_group"] == torch.arange(self.V, device=st["tok_group"].device)
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
        """Rank cut of the key (value) group of head h: the members' query factors are stacked with
        block-diagonal weights (their output maps side by side with pooled weights) and cut together
        with the group's keys (values).  For a group of one this is her qk_rank / ov_rank."""
        st = s.struct
        Lc = st["stream_alive"].nonzero().flatten()
        nL = len(Lc)
        x, sc = c["x"][2 * l][..., Lc].double(), c["s"][2 * l].double()
        E = lambda n, hh: s.params[n][l, hh] * s.masks[n][l, hh].to(s.params[n].dtype)
        feats = {"pQ": 1 / sc, "pK": 1 / sc, "pV": 1 / sc, "b_Q": torch.ones_like(sc)}
        if kind == "qk_rank":
            g = int(st["k_group"][l, h])
            mem = self.k_members(s, l, g)
            used = torch.zeros(self.dh, dtype=torch.bool, device=Lc.device)
            for m in mem:
                used |= self.qk_dims(s, l, m)
            r = int(used.sum()) - 1
            if r < 1 or not mem:
                return False
            inc_k = [n for n in ("pK",) if bool(s.masks[n][l, g].any())]
            ak = torch.cat([x / sc] + [feats[n] for n in inc_k], -1).reshape(-1, nL + len(inc_k))
            WK = torch.cat([E("W_K", g)[Lc]] + [E(n, g)[None] for n in inc_k]).double()
            WQs, Cqs, incs = [], [], []
            for m in mem:
                inc_q = [n for n in ("pQ", "b_Q") if bool(s.masks[n][l, m].any())]
                aq = torch.cat([x / sc] + [feats[n] for n in inc_q], -1).reshape(-1, nL + len(inc_q))
                WQs.append(torch.cat([E("W_Q", m)[Lc]] + [E(n, m)[None] for n in inc_q]).double())
                Cqs.append(aq.T @ aq / len(aq)); incs.append(inc_q)
            A, B = weighted_lowrank(torch.cat(WQs) @ WK.T, torch.block_diag(*Cqs), ak.T @ ak / len(ak), r)
            i0 = 0
            for m, Wm, inc_q in zip(mem, WQs, incs):
                self._write_factor(s, "W_Q", inc_q, l, m, Lc, A[i0:i0 + len(Wm)], r)
                i0 += len(Wm)
            self._write_factor(s, "W_K", inc_k, l, g, Lc, B, r)
            return True
        g = int(st["v_group"][l, h])
        mem = self.v_members(s, l, g)
        used = torch.zeros(self.dh, dtype=torch.bool, device=Lc.device)
        for m in mem:
            used |= self.ov_dims(s, l, m)
        r = int(used.sum()) - 1
        if r < 1 or not mem:
            return False
        inc_v = [n for n in ("pV",) if bool(s.masks[n][l, g].any())]
        av = torch.cat([x / sc] + [feats[n] for n in inc_v], -1)                        # (n, T, nL + .)
        Cy = 0
        for m in mem:
            y = torch.einsum("nts,nsk->ntk", c["A"][l][:, m].double(), av).reshape(-1, av.shape[-1])
            Cy = Cy + y.T @ y / len(y)
        WV = torch.cat([E("W_V", g)[Lc]] + [E(n, g)[None] for n in inc_v]).double()
        N = WV @ torch.cat([E("W_O", m)[:, Lc].double() for m in mem], 1)
        A, B = weighted_lowrank(N, Cy, torch.eye(nL * len(mem), dtype=torch.double,
                                                 device=N.device), r)
        self._write_factor(s, "W_V", inc_v, l, g, Lc, A, r)
        rows = torch.arange(r, device=Lc.device)
        for i, m in enumerate(mem):
            self._zero(s, "W_O", (l, m))
            s.params["W_O"][l, m][rows[:, None], Lc[None, :]] = B[i * nL:(i + 1) * nL].T
            s.masks["W_O"][l, m][rows[:, None], Lc[None, :]] = True
        return True

    # ---- key / value sharing ------------------------------------------------------------------------------
    def _alone(self, s, l, h, which):
        """h is the only user of its own key (value) slot."""
        st = s.struct
        grp = st["k_group" if which == "k" else "v_group"][l]
        mem = self.k_members(s, l, h) if which == "k" else self.v_members(s, l, h)
        return int(grp[h]) == h and mem == [h]

    @torch.no_grad()
    def _k_share(self, s, c, l, h, g):
        st = s.struct
        if h == g or not self._alone(s, l, h, "k") or not self.k_members(s, l, g):
            return False
        Lc = st["stream_alive"].nonzero().flatten()
        nL = len(Lc)
        x, sc = c["x"][2 * l][..., Lc].double(), c["s"][2 * l].double()
        E = lambda n, hh: (s.params[n][l, hh] * s.masks[n][l, hh].to(s.params[n].dtype)).double()
        inc_k = bool(s.masks["pK"][l, h].any() or s.masks["pK"][l, g].any())
        ak = torch.cat([x / sc] + ([1 / sc] if inc_k else []), -1).reshape(-1, nL + int(inc_k))
        Ck = ak.T @ ak / len(ak)
        K = lambda hh: torch.cat([E("W_K", hh)[Lc]] + ([E("pK", hh)[None]] if inc_k else []))
        inc_q = [n for n in ("pQ", "b_Q") if bool(s.masks[n][l, h].any())]
        WQ = torch.cat([E("W_Q", h)[Lc]] + [E(n, h)[None] for n in inc_q])
        kd = self._k_slot(s, l, g).nonzero().flatten()
        Kg = K(g)[:, kd]
        G_ = Kg.T @ Ck @ Kg
        Qn = (WQ @ K(h).T) @ Ck @ Kg @ torch.linalg.pinv(G_, hermitian=True)          # (dq, |kd|)
        for n in ["W_Q"] + inc_q:
            self._zero(s, n, (l, h))
        s.params["W_Q"][l, h][Lc[:, None], kd[None, :]] = Qn[:nL].float()
        s.masks["W_Q"][l, h][Lc[:, None], kd[None, :]] = True
        for i, n in enumerate(inc_q):
            s.params[n][l, h][kd] = Qn[nL + i].float()
            s.masks[n][l, h][kd] = True
        st["k_group"][l, h] = g
        return True

    @torch.no_grad()
    def _v_share(self, s, c, l, h, g):
        st = s.struct
        if h == g or not self._alone(s, l, h, "v") or not self.v_members(s, l, g):
            return False
        Lc = st["stream_alive"].nonzero().flatten()
        nL = len(Lc)
        x, sc = c["x"][2 * l][..., Lc].double(), c["s"][2 * l].double()
        E = lambda n, hh: (s.params[n][l, hh] * s.masks[n][l, hh].to(s.params[n].dtype)).double()
        inc_v = bool(s.masks["pV"][l, h].any() or s.masks["pV"][l, g].any())
        av = torch.cat([x / sc] + ([1 / sc] if inc_v else []), -1)
        y = torch.einsum("nts,nsk->ntk", c["A"][l][:, h].double(), av).reshape(-1, av.shape[-1])
        Cy = y.T @ y / len(y)
        V = lambda hh: torch.cat([E("W_V", hh)[Lc]] + ([E("pV", hh)[None]] if inc_v else []))
        vd = self._v_slot(s, l, g).nonzero().flatten()
        Vg = V(g)[:, vd]
        Nh = V(h) @ E("W_O", h)[:, Lc]
        On = torch.linalg.pinv(Vg.T @ Cy @ Vg, hermitian=True) @ Vg.T @ Cy @ Nh         # (|vd|, nL)
        self._zero(s, "W_O", (l, h))
        s.params["W_O"][l, h][vd[:, None], Lc[None, :]] = On.float()
        s.masks["W_O"][l, h][vd[:, None], Lc[None, :]] = True
        st["v_group"][l, h] = g
        return True

    def _write_factor(self, s, W, inc, l, h, Lc, F_, r):
        for n in [W] + inc:
            self._zero(s, n, (l, h))
        cols = torch.arange(r, device=Lc.device)
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
            rows, cols = (s.struct["tok_group"] == torch.arange(
                self.V, device=Lc.device)).nonzero().flatten(), Lc
        elif kind == "pos_rank":
            rows, cols = torch.arange(self.T, device=Lc.device), Lc
        else:
            rows, cols = Lc, torch.arange(self.V, device=Lc.device)
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
            Cl = torch.eye(len(rows), dtype=torch.double, device=W.device)
        else:
            y = (c["x"][-1][..., rows] / c["s"][-1]).reshape(-1, len(rows)).double()
            Cl = y.T @ y / len(y)
        A, B = weighted_lowrank(W, Cl, torch.eye(len(cols), dtype=torch.double,
                                                 device=W.device), r)
        for n in (dense, a, b):
            if bool(s.masks[n].any()):
                zero_entries(s, n, s.masks[n].clone())
        k = torch.arange(r, device=W.device)
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
        if not all(self._alone(s, l, hh, w) for hh in (h, h2) for w in ("k", "v")):
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
        A, B = weighted_lowrank(N, y.T @ y / len(y), torch.eye(nL, dtype=torch.double,
                                                               device=N.device), r)
        self._write_factor(s, "W_V", ["pV"] if inc else [], l, h2, Lc, A, r)
        self._zero(s, "W_O", (l, h2))
        k = torch.arange(r, device=Lc.device)
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
            iu = torch.triu_indices(len(ids), len(ids), 1, device=ids.device)
            vals = Cr[iu[0], iu[1]].abs()
            for t_ in torch.argsort(vals, descending=True)[: cfg.n_merge].tolist():
                a_, b_ = int(ids[iu[0, t_]]), int(ids[iu[1, t_]])
                pairs += [(l, a_, b_), (l, b_, a_)]
        g = s.struct["tok_group"]
        reps = (g == torch.arange(self.V, device=g.device)).nonzero().flatten()
        E = self.effective(s)
        Em = (E["W_E"] + E["E_a"] @ E["E_b"])[reps].double()
        f = torch.tensor([self._group_freq(s, int(i)) for i in reps], dtype=torch.double,
                         device=g.device)
        cost = (f[:, None] * f[None, :] / (f[:, None] + f[None, :]).clamp_min(1e-30)) * torch.cdist(Em, Em).pow(2)
        iu = torch.triu_indices(len(reps), len(reps), 1, device=reps.device)
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
                kmem = self.k_members(s, l, int(st["k_group"][l, h]))
                vmem = self.v_members(s, l, int(st["v_group"][l, h]))
                if not bool(st["head_fixed"][l, h]) and qk > 0:
                    mv.append(Move("attn_uniform", site, l, {"l": l, "h": h}))
                    mv.append(Move("attn_fixed", site, l, {"l": l, "h": h}))
                    if int(self._k_slot(s, l, int(st["k_group"][l, h])).sum()) > 1 and kmem and kmem[0] == h:
                        mv.append(Move("qk_rank", site, l, {"l": l, "h": h}))
                if int(self._v_slot(s, l, int(st["v_group"][l, h])).sum()) > 1 and vmem and vmem[0] == h:
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
            for h in live:
                for g in range(self.H):
                    if g == h:
                        continue
                    if self._alone(s, l, h, "k") and self.k_members(s, l, g):
                        mv.append(Move("k_share", f"L{l}h{h}", l, {"l": l, "h": h, "g": g}))
                    if self._alone(s, l, h, "v") and self.v_members(s, l, g):
                        mv.append(Move("v_share", f"L{l}h{h}", l, {"l": l, "h": h, "g": g}))
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
        if m.kind == "k_share":
            return self._k_share(s, c, p["l"], p["h"], p["g"])
        if m.kind == "v_share":
            return self._v_share(s, c, p["l"], p["h"], p["g"])
        l = p["l"]
        if m.kind.startswith("unit_"):
            i = p["i"]
            if int(st["unit_type"][l, i]) != ACTIVE:
                return False
            if m.kind == "unit_sat":
                self._kill_unit(s, l, i)
                return True
            if m.kind == "unit_mean":
                mean = p["mean"] if "mean" in p else c["act"][l][..., i].mean()
                self._fold_unit(s, l, i, float(mean))
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
            mean = p["mean"] if "mean" in p else c["ho"][l][:, h].mean((0, 1))
            s.params["b_O"][l] += mean
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


def load(L=2, seed=0, root=None, device="cpu", **kw):
    if root is None and (L, seed) == (2, 0):
        path = os.path.join(os.path.dirname(HERE), "checkpoints", "gpt_l2_seed0.pt")
    else:
        path = os.path.join(root or os.path.join(HERE, "results_gpt"),
                            f"L{L}_seed{seed}", "base_state.pt")
    base = torch.load(path, map_location="cpu", weights_only=False)
    ad = GPTAdapter(make_data(seed), G.GPTConfig(**base["cfg"]), **kw)
    state = ad.stage1(base["params"])
    return ad.to(device), state.to(device), base
