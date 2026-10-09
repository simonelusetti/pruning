"""Checks for the low-rank conv move.  Run: python -m pytest -q test_cnn_fm_rank.py"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_fm import load_fashion, train_epochs, _REG  # noqa: E402
from cnn_fm_rank import CNNAdapterFMRank, RANK_KINDS  # noqa: E402
from nnprune.core import Move, PruneConfig, REMOVED, prune_path, zero_entries  # noqa: E402

torch.manual_seed(0)
DATA, FIT = load_fashion(0)
DATA = {k: (v[0][:n], v[1][:n]) for (k, v), n in zip(DATA.items(), (500, 600, 300, 300, 1000))}
AD = CNNAdapterFMRank(DATA, stats_n=300)
_ck = "/tmp/_fmr_test_train.pt"
if os.path.exists(_ck):
    os.remove(_ck)
S0 = train_epochs(AD, (FIT[0][:20000], FIT[1][:20000]), _ck, seed=0, epochs=2)
os.remove(_ck)
X = DATA["train"][0][:300]


def logits(s):
    return AD.logits(AD.effective(s), s.struct, X)


def lowrank(s, r):
    p = AD._rank_moves(s, [r])[0]
    return AD.apply(s, Move("conv_lowrank", "c2w", 1, p))


def test_full_rank_factorisation_is_exact():
    s = lowrank(S0, 32)
    assert int(s.struct["fact"][0]) == 1
    assert torch.allclose(logits(s), logits(S0), atol=1e-3)


def test_raw_and_effective_parameters_agree():
    s = lowrank(S0, 8)
    raw = {k: v * s.masks[k].to(v.dtype) for k, v in s.params.items()}
    assert torch.allclose(AD.logits(raw, s.struct, X), logits(s), atol=1e-4)


def test_parameter_count_and_mean_preserved():
    s = lowrank(S0, 16)
    assert AD.k_eff(s) == AD.k_eff(S0) - 32 * 16 * 9 + 16 * (16 * 9 + 32)
    P0, P1 = AD.effective(S0), AD.effective(s)
    a1 = AD.conv1_out(P0, S0.struct, X)
    z0 = F.conv2d(a1, P0["W2"], P0["b2"], padding=1)
    z1 = F.conv2d(a1, P1["W2"], P1["b2"], padding=1)
    assert (z1 - z0).mean((0, 2, 3)).abs().max() < 1e-4


def test_rank_cut_and_component_cascade():
    s = lowrank(S0, 12)
    p = AD._rank_moves(s, [11])[0]
    s2 = AD.apply(s, Move("conv_rankcut", "c2w", 1, p))
    assert AD.k_eff(s2) == AD.k_eff(s) - (16 * 9 + 32)
    s3 = s2.clone()
    col = torch.zeros_like(s3.masks["W2b"]); col[:, 3] = True
    zero_entries(s3, "W2b", col)
    before = logits(s3)
    AD.canonicalize(s3)
    assert not bool(s3.masks["W2a"][3].any())
    assert torch.allclose(logits(s3), before, atol=1e-5)


def test_pointwise_merge_on_factorised_matches_dense():
    s = lowrank(S0, 12)
    dense = AD.make_state({k: v for k, v in AD.effective(s).items()})   # same function, unfactorised
    cfg = PruneConfig(n_merge=4, verbose=False)
    ctx = AD.refresh_extra(s, cfg)
    m = [mm for mm in AD.structural_moves(s, ctx, cfg) if mm.kind == "chan_merge_pw"][0]
    assert torch.allclose(logits(AD.apply(s, m)), logits(AD.apply(dense, m)), atol=1e-4)


def test_channel_removal_cascades_when_factorised():
    s = lowrank(S0, 12)
    col = torch.zeros_like(s.masks["W2a"]); col[:, 5] = True
    zero_entries(s, "W2a", col)
    AD.canonicalize(s)
    assert int(s.struct["type0"][5]) == REMOVED


def test_path_runs_with_rank_moves():
    cfg = PruneConfig(n_merge=4, verbose=False, screen_n=300, max_steps=4, weight_per_site=0, n_struct=8,
                      struct_kinds=RANK_KINDS + ("gap_merge",))
    df, s = prune_path(AD, S0, cfg)
    assert df["k_eff"].is_monotonic_decreasing and len(df) > 1
