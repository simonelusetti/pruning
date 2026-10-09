"""Checks for the stream-direction move.  Run: python -m pytest -q test_tf_stream.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "NNpruning"))
from nnprune import transformer as T  # noqa: E402
from nnprune.core import Move, PruneConfig, prune_path  # noqa: E402
from nnprune.experiments import split_validation  # noqa: E402
from tf_stream import StreamAdapter, STREAM_KINDS  # noqa: E402
import stream_directions as SD  # noqa: E402

DATA = split_validation(T.make_modadd_data(p=59, train_frac=0.5, seed=0), 0)
HER = T.TransformerAdapter(DATA, p=59)
AD = StreamAdapter(DATA, p=59)
S_HER = torch.load(os.path.join(os.path.dirname(HERE), "NNpruning", "results", "transformer",
                                "base_state_seed0.pt"), weights_only=False)
S0 = AD.upgrade(S_HER)
X = DATA["val"][0]
CFG = PruneConfig(verbose=False)


def logits(ad, s):
    return ad.logits(ad.effective(s), s.struct, X)


def first_cand(s):
    return AD.refresh_extra(s, CFG)["stream_cands"][0]


def test_upgrade_is_the_same_function():
    assert torch.equal(logits(AD, S0), logits(HER, S_HER))
    assert AD.k_eff(S0) == HER.k_eff(S_HER) == 57339


def test_reflection_is_exact():
    s = S0.clone()
    u = torch.randn(64, dtype=torch.double); u /= u.norm()
    AD._rotate(s, AD._householder(u, 63, 64))
    assert torch.allclose(logits(AD, s), logits(AD, S0), atol=1e-4)
    assert AD.k_eff(s) == AD.k_eff(S0)


def test_move_frees_891_minus_new_biases():
    s = AD.apply(S0, Move("stream_mean", "stream", 0, {"u": first_cand(S0)}))
    assert AD.k_eff(S0) - AD.k_eff(s) == 891 - 448
    s2 = AD.apply(s, Move("stream_mean", "stream", 0, {"u": first_cand(s)}))
    assert AD.k_eff(s) - AD.k_eff(s2) == 891                # biases already counted


def test_move_matches_the_projection_of_stream_directions():
    """One stream move along u == x -> Pi x + (I - Pi) m at every use (stream_directions.py)."""
    u = first_cand(S0)
    s = AD.apply(S0, Move("stream_mean", "stream", 0, {"u": u}))
    ev = SD.Evaluator(HER, S_HER, DATA)
    Pi = torch.eye(64) - torch.outer(u, u)
    for split in ("val", "test"):
        assert abs(AD.metrics(s, split)["loss"] - ev.run([Pi] * 3, split)[0]) < 1e-6


def test_rank_move_after_stream_move_uses_rotated_statistics():
    ctx = AD.refresh_extra(S0, CFG)
    s = AD.apply(S0, Move("stream_mean", "stream", 0, {"u": ctx["stream_cands"][0]}))
    m = [mm for mm in AD.structural_moves(S0, ctx, CFG) if mm.kind == "qk_rank"][0]
    a = AD.apply(s, m)                                         # stale ctx, rotated inside
    ctx2 = AD.refresh_extra(s, CFG)
    m2 = [mm for mm in AD.structural_moves(s, ctx2, CFG) if mm.kind == "qk_rank" and mm.site == m.site][0]
    b = AD.apply(s, m2)                                        # fresh ctx
    assert (AD.metrics(a, "val")["loss"] - AD.metrics(b, "val")["loss"]) < 1e-3


def test_cascade_dead_reader():
    s = S0.clone()
    j = 5
    for name, ax in (("W_Q", 1), ("W_K", 1), ("W_V", 1), ("W_in", 1), ("W_U", 0)):
        mk = torch.zeros_like(s.masks[name]); mk.select(ax, j).fill_(True)
        s.masks[name][mk] = False; s.params[name][mk] = 0
    AD.canonicalize(s)
    assert not bool(s.struct["stream_alive"][j]) and not bool(s.masks["W_E"][:, j].any())


def test_path_runs():
    cfg = PruneConfig(verbose=False, max_steps=6, weight_per_site=0, n_struct=12, screen_n=256,
                      refresh_every=3, struct_kinds=STREAM_KINDS + ("unit_merge", "qk_rank"),
                      select_split="valA")
    df, s = prune_path(AD, S0, cfg)
    assert df["k_eff"].is_monotonic_decreasing and len(df) > 1


def test_weight_zero_after_stream_move():
    """single-weight moves index parameters flat: tensors must stay contiguous."""
    from nnprune.core import zero_entries
    s = AD.apply(S0, Move("stream_mean", "stream", 0, {"u": first_cand(S0)}))
    for name in ("W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_E", "W_U"):
        live = s.masks[name].reshape(-1).nonzero().flatten()[:3]
        zero_entries(s, name, live)


def _after_stream(n=2):
    s = S0
    for _ in range(n):
        s = AD.apply(s, Move("stream_mean", "stream", 0, {"u": first_cand(s)}))
    return s


def test_qk_cut_after_stream_move_is_still_exact():
    """QK cuts are exact nulls in her model (constant query input); with live q/k biases
    the augmented cut must keep them exact."""
    s = _after_stream()
    ctx = AD.refresh_extra(s, CFG)
    m = [mm for mm in AD.structural_moves(s, ctx, CFG) if mm.kind == "qk_rank" and mm.payload["h"] == 0][0]
    s2 = AD.apply(s, m)
    assert abs(AD.metrics(s2, "val")["loss"] - AD.metrics(s, "val")["loss"]) < 1e-6
    assert AD.k_eff(s) - AD.k_eff(s2) == 2 * (int(s.struct["stream_alive"].sum())) + 1 + 3


def test_ov_cut_after_stream_move_is_small():
    s = _after_stream()
    ctx = AD.refresh_extra(s, CFG)
    m = [mm for mm in AD.structural_moves(s, ctx, CFG) if mm.kind == "ov_rank" and mm.payload["h"] == 0][0]
    s2 = AD.apply(s, m)
    m0 = [mm for mm in HER.structural_moves(S_HER, HER.refresh_extra(S_HER, CFG), CFG)
          if mm.kind == "ov_rank" and mm.payload["h"] == 0][0]
    her_cost = HER.metrics(HER.apply(S_HER, m0), "val")["loss"] - HER.metrics(S_HER, "val")["loss"]
    cost = AD.metrics(s2, "val")["loss"] - AD.metrics(s, "val")["loss"]
    assert cost < 10 * abs(her_cost) + 1e-5


def test_dead_coordinates_stay_dead_after_rank_cuts():
    s = _after_stream()
    ctx = AD.refresh_extra(s, CFG)
    for mm in AD.structural_moves(s, ctx, CFG):
        if mm.kind in ("qk_rank", "ov_rank"):
            s = AD.apply(s, mm) or s
    from tf_stream import ADDED, READ
    for j in (~s.struct["stream_alive"]).nonzero().flatten().tolist():
        for name, ax in list(ADDED.items()) + list(READ.items()):
            assert not bool(s.masks[name].select(ax, j).any())


def test_fresh_slots_never_go_stale():
    ad = StreamAdapter(DATA, p=59, fresh=True)
    s = ad.upgrade(S_HER)
    ctx = ad.refresh_extra(s, CFG)
    mv = [m for m in ad.structural_moves(s, ctx, CFG) if m.kind == "stream_mean"]
    assert all("slot" in m.payload for m in mv)
    for _ in range(3):                                  # the same slot keeps giving a valid, fresh move
        s2 = ad.apply(s, mv[0])
        assert s2 is not None and ad.k_eff(s) - ad.k_eff(s2) in (891, 891 - 448)
        s = s2
    # slot 0 applied now == explicit u from the current candidate list
    u = ad.stream_candidates(s)[0]
    a = ad.apply(s, mv[0]); b = ad.apply(s, Move("stream_mean", "stream", 0, {"u": u}))
    assert torch.allclose(ad.logits(ad.effective(a), a.struct, X), ad.logits(ad.effective(b), b.struct, X))
