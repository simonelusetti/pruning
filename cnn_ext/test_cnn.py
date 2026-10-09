"""Checks for the CNN adapter.  Run: python -m pytest -q test_cnn.py  (from this folder)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn import CNNAdapter, make_digits_data, train_cnn, _REG  # noqa: E402
from nnprune.core import PruneConfig, Move, REMOVED, zero_entries, prune_path  # noqa: E402

torch.manual_seed(0)
AD = CNNAdapter(make_digits_data(0))
S0 = train_cnn(AD, seed=0, steps=300)
X, _ = AD.data["train"]
CFG = PruneConfig(n_merge=8, verbose=False)
CTX = AD.refresh_extra(S0, CFG)


def z2(s):
    P = AD.effective(s)
    a1 = F.relu(F.conv2d(X, P["W1"], P["b1"], padding=1))
    a1 = a1 * (s.struct["type0"] != REMOVED).float()[None, :, None, None]
    return F.conv2d(a1, P["W2"], P["b2"], padding=1) + P["B2"][:, _REG][:, :, _REG]


def logits(s):
    return AD.logits(AD.effective(s), s.struct, X)


def test_fanin_zero_folds_exactly_conv1():
    j = 3
    s = S0.clone()
    zero_entries(s, "W1", torch.arange(9) + j * 9)
    before = logits(s)                      # channel j is now the constant relu(b1[j])
    k_before = AD.k_eff(s)
    AD.canonicalize(s)
    assert int(s.struct["type0"][j]) == REMOVED
    assert torch.allclose(logits(s), before, atol=1e-4)
    # freed: b1 (1) + W2[:, j] (32*9) ; B2 activated (+9*32) if relu(b1) > 0
    assert AD.k_eff(s) < k_before + 9 * 32


def test_fanin_zero_folds_exactly_conv2():
    o = 5
    s = S0.clone()
    row = torch.zeros_like(s.masks["W2"]); row[o] = True
    zero_entries(s, "W2", row)
    before = logits(s)
    AD.canonicalize(s)
    assert int(s.struct["type1"][o]) == REMOVED
    assert torch.allclose(logits(s), before, atol=1e-4)


def test_fanout_zero_removes_channel():
    j = 2
    s = S0.clone()
    col = torch.zeros_like(s.masks["W2"]); col[:, j] = True
    zero_entries(s, "W2", col)
    before = logits(s)
    AD.canonicalize(s)
    assert int(s.struct["type0"][j]) == REMOVED
    assert torch.allclose(logits(s), before, atol=1e-5)
    assert AD.k_eff(s) == AD.k_eff(S0) - 32 * 9 - 10


def _move(kind):
    return [m for m in AD.structural_moves(S0, CTX, CFG) if m.kind == kind][0]


def test_output_side_scalar_is_unbiased():
    m = _move("chan_mean_s")
    s = AD.apply(S0, m)
    d = (z2(s) - z2(S0)).mean((0, 2, 3))       # mean change of every conv2 pre-activation
    assert d.abs().max() < 1e-4


def test_output_side_region_is_unbiased_per_region_and_counted_once():
    m = _move("chan_mean_r")
    s = AD.apply(S0, m)
    d = (z2(s) - z2(S0)).mean(0)                # (C2, 8, 8)
    for r in range(3):
        for c in range(3):
            assert d[:, _REG == r][:, :, _REG == c].mean((1, 2)).abs().max() < 1e-4
    assert AD.k_eff(s) == AD.k_eff(S0) - (9 + 1 + 32 * 9) + 9 * 32
    m2 = [mm for mm in AD.structural_moves(S0, CTX, CFG) if mm.kind == "chan_mean_r" and mm.payload["j"] != m.payload["j"]][0]
    s2 = AD.apply(s, m2)
    assert AD.k_eff(s2) == AD.k_eff(s) - (9 + 1 + 32 * 9)      # B2 already paid for


def test_spatial_merge_is_unbiased_and_keeps_kernel_count():
    m = _move("chan_merge_sp")
    s = AD.apply(S0, m)
    d = (z2(s) - z2(S0)).mean((0, 2, 3))
    assert d.abs().max() < 1e-3
    assert AD.k_eff(s) == AD.k_eff(S0) - (9 + 1 + 32 * 9)


def test_gap_mean_exact_on_constant():
    m = _move("gap_mean")
    s = AD.apply(S0, m)
    o = m.payload["o"]
    P = AD.effective(S0)
    _, _, g = AD.logits(P, S0.struct, X, return_acts=True)
    g2 = g.clone(); g2[:, o] = m.payload["g"]
    assert torch.allclose(logits(s), g2 @ P["Wf"].T + P["bf"], atol=1e-4)


def test_path_runs():
    cfg = PruneConfig(n_merge=8, max_steps=3, verbose=False, weight_per_site=2, n_struct=8)
    df, s = prune_path(AD, S0, cfg)
    assert len(df) == 4 and df["k_eff"].is_monotonic_decreasing
