"""Checks for gpt_adapter.py.  Run: python3 test_gpt_adapter.py  (or pytest)"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpt_adapter as GA  # noqa: E402
import gpt_toy as G  # noqa: E402
from nnprune.core import Move, PruneConfig, prune_path  # noqa: E402

torch.set_num_threads(4)
AD, S0, BASE = GA.load(L=2, seed=0)
X = AD.data["val"][0][:256]
CFG = PruneConfig(verbose=False, n_merge=4)


def logits(s, X=X):
    return AD.logits(AD.effective(s), s.struct, X)


def test_stage1_is_exact():
    z0 = G.forward(BASE["params"], AD.cfg, X)
    assert float((logits(S0) - z0).abs().max()) < 1e-4 * float(z0.abs().max())


def test_stage1_count():
    """gamma, beta: 5 x 128; b_K, b_V: 2 x 64 each; the direction 1: one stream coordinate
    (writers: W_E 65 + W_pos 32 + per layer W_O 64 + b_O 1 + W_2 256 + b_2 1; readers: per
    layer W_Q, W_K, W_V 3 x 64 + W_1 256; W_U 65)."""
    p = G.n_params(BASE["params"])
    per_coord = 65 + 32 + 2 * (64 + 1 + 256 + 1) + 2 * (3 * 64 + 256) + 65
    assert per_coord == 1702
    assert p - AD.k_eff(S0) == 5 * 128 + 2 * 64 + 2 * 64 + per_coord


def test_rotation_is_exact():
    s = S0.clone()
    u = torch.randn(64, dtype=torch.double); u[~s.struct["stream_alive"]] = 0; u /= u.norm()
    AD._rotate(s, GA.householder(u, 10, 64))
    assert torch.allclose(logits(s), logits(S0), atol=1e-4)
    assert AD.k_eff(s) == AD.k_eff(S0)


def replaced_forward(s, u, X, Xstats):
    """The network of state s with u.x replaced by its training mean m_k at the input of
    every LayerNorm k (means computed depth by depth on Xstats)."""
    P, S = AD.effective(s), s.struct
    u = u.float()

    def run(X, ms):
        T = X.shape[1]
        causal = AD.causal[:T, :T]
        x = P["W_E"][X] + P["W_pos"][:T]
        out = []

        def site(x, k):
            m = float((x @ u).mean()) if k >= len(ms) else ms[k]
            if k >= len(ms):
                ms.append(m)
            xr = x - (x @ u)[..., None] * u + m * u
            return xr, torch.sqrt((xr.pow(2).sum(-1, keepdim=True) + P["M"][k]) / 64 + G.EPS)

        for l in range(AD.L):
            xr, s1 = site(x, 2 * l)
            sh = s1[:, None]
            q = (torch.einsum("btd,hde->bhte", xr, P["W_Q"][l]) + P["pQ"][l][:, None]) / sh + P["b_Q"][l][:, None]
            k = (torch.einsum("btd,hde->bhte", xr, P["W_K"][l]) + P["pK"][l][:, None]) / sh
            v = (torch.einsum("btd,hde->bhte", xr, P["W_V"][l]) + P["pV"][l][:, None]) / sh
            A = torch.softmax((torch.einsum("bhte,bhse->bhts", q, k) / 4).masked_fill(~causal, float("-inf")), -1)
            x = x + torch.einsum("bhts,bhse,hed->btd", A, v, P["W_O"][l]) + P["b_O"][l] + P["B_attn"][l]
            xr, s2 = site(x, 2 * l + 1)
            x = x + F.gelu((xr @ P["W_1"][l].T + P["p1"][l]) / s2 + P["b_1"][l]) @ P["W_2"][l].T + P["b_2"][l]
        xr, sf = site(x, 2 * AD.L)
        return (xr @ P["W_U"] + P["pU"]) / sf + P["b_U"]

    ms = []
    run(Xstats, ms)
    return run(X, ms)


def test_stream_move_equals_replaced_network():
    for s in (S0, AD.apply(S0, Move("stream_mean", "stream", 0, {"slot": 0}))):   # first and second move
        c = AD._stats(s)
        u = AD.stream_candidates(s, c)[0]
        s2 = AD.apply(s, Move("stream_mean", "stream", 0, {"slot": 0}))
        ref = replaced_forward(s, u, X, AD.data["stats"][0])
        assert float((logits(s2) - ref).abs().max()) < 1e-4 * float(ref.abs().max())


def test_stream_move_counts():
    s1 = AD.apply(S0, Move("stream_mean", "stream", 0, {"slot": 0}))
    s2 = AD.apply(s1, Move("stream_mean", "stream", 0, {"slot": 0}))
    # pre-division biases (2 x (3 x 64 + 256) + 65 = 961) and 5 numbers M_k are paid once
    assert AD.k_eff(S0) - AD.k_eff(s1) == 1702 - 961 - 5
    assert AD.k_eff(s1) - AD.k_eff(s2) == 1702


def test_every_move_kind_applies():
    ctx = AD.refresh_extra(S0, CFG)
    seen = set()
    for m in AD.structural_moves(S0, ctx, CFG):
        if m.kind in seen:
            continue
        s2 = AD.apply(S0, m)
        assert s2 is not None, m.kind
        assert AD.k_eff(s2) < AD.k_eff(S0), m.kind
        assert all(bool(torch.isfinite(v).all()) for v in s2.params.values())
        assert math.isfinite(AD.metrics(s2, "valA")["loss"])
        seen.add(m.kind)
    assert seen == set(GA.HER_KINDS + GA.STREAM_KINDS)


def test_rank_cuts_after_stream_move():
    s = AD.apply(S0, Move("stream_mean", "stream", 0, {"slot": 0}))
    ctx = AD.refresh_extra(s, CFG)
    for m in AD.structural_moves(s, ctx, CFG):
        if m.kind in ("qk_rank", "ov_rank") and m.payload["h"] == 0:
            s2 = AD.apply(s, m)
            dead = ~s2.struct["stream_alive"]
            for n, ax in GA.STREAM_AX.items():
                assert not bool((GA._along(s2.masks[n], ax) & dead).any()), n
            d_loss = AD.metrics(s2, "valA")["loss"] - AD.metrics(s, "valA")["loss"]
            assert -0.05 < d_loss < 0.5, (m.kind, d_loss)


def test_canonicalize_is_idempotent():
    s = AD.apply(S0, Move("head_off", "L1h0", 1, {"l": 1, "h": 0}))
    k = AD.k_eff(s)
    AD.canonicalize(s)
    assert AD.k_eff(s) == k


def test_path_runs():
    cfg = PruneConfig(verbose=False, max_steps=3, weight_per_site=1, weight_chunk=64, weight_score="magnitude",
                      n_struct=6, screen_n=64, refresh_every=2, n_merge=2, select_split="valA",
                      stop_split="valB", stop_mode="mark", select_admissibility=False, tie_break="k_eff",
                      post_stop_steps=10)
    df, _ = prune_path(AD, S0, cfg)
    assert df["k_eff"].is_monotonic_decreasing and len(df) == 4


if __name__ == "__main__":
    import time
    for n in [n for n in dir() if n.startswith("test_")]:
        t = time.time(); globals()[n](); print(f"ok {n} ({time.time() - t:.1f}s)", flush=True)
