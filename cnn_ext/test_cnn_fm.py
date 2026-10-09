"""Checks for the Fashion-MNIST adapter and the resumable path.
Run: python -m pytest -q test_cnn_fm.py  (from this folder; needs ../data/fashion)."""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_fm import CNNAdapterFM, load_fashion, train_epochs, _REG  # noqa: E402
from nnprune.core import PruneConfig, REMOVED, zero_entries, prune_path  # noqa: E402
from resumable import prune_path_resumable  # noqa: E402

torch.manual_seed(0)
DATA, FIT = load_fashion(0)
DATA = {k: (v[0][:n], v[1][:n]) for (k, v), n in zip(DATA.items(), (500, 600, 300, 300, 1000))}  # small, for speed
AD = CNNAdapterFM(DATA, stats_n=500)
_ck = "/tmp/_fm_test_train.pt"
if os.path.exists(_ck):
    os.remove(_ck)
S0 = train_epochs(AD, (FIT[0][:20000], FIT[1][:20000]), _ck, seed=0, epochs=2)
os.remove(_ck)
X = DATA["train"][0][:500]
CFG = PruneConfig(n_merge=4, verbose=False, screen_n=500)
CTX = AD.refresh_extra(S0, CFG)


def logits(s):
    return AD.logits(AD.effective(s), s.struct, X)


def z2(s):
    P = AD.effective(s)
    a1 = AD.conv1_out(P, s.struct, X)
    return F.conv2d(a1, P["W2"], P["b2"], padding=1) + P["B2"][:, _REG][:, :, _REG]


def test_trained_better_than_chance():
    assert AD.metrics(S0, "test")["acc"] > 0.5


def test_fanin_zero_folds_exactly_both_layers():
    s = S0.clone()
    zero_entries(s, "W1", torch.arange(9) + 3 * 9)
    before = logits(s)
    AD.canonicalize(s)
    assert int(s.struct["type0"][3]) == REMOVED
    assert torch.allclose(logits(s), before, atol=1e-4)
    s = S0.clone()
    row = torch.zeros_like(s.masks["W2"]); row[5] = True
    zero_entries(s, "W2", row)
    before = logits(s)
    AD.canonicalize(s)
    assert int(s.struct["type1"][5]) == REMOVED
    assert torch.allclose(logits(s), before, atol=1e-4)


def _move(kind):
    return [m for m in AD.structural_moves(S0, CTX, CFG) if m.kind == kind][0]


def test_output_side_folds_unbiased():
    for kind in ("chan_mean_s", "chan_merge_sp"):
        s = AD.apply(S0, _move(kind))
        assert (z2(s) - z2(S0)).mean((0, 2, 3)).abs().max() < 1e-3, kind
    s = AD.apply(S0, _move("chan_mean_r"))
    d = (z2(s) - z2(S0)).mean(0)
    for r in range(3):
        for c in range(3):
            assert d[:, _REG == r][:, :, _REG == c].mean((1, 2)).abs().max() < 1e-3


def test_resumable_matches_uninterrupted():
    cfg = PruneConfig(n_merge=4, verbose=False, screen_n=500, max_steps=6, weight_per_site=2, n_struct=6,
                      weight_chunk=8, refresh_every=3, select_split="valA", stop_split="valB",
                      select_admissibility=False, stop_mode="mark", post_stop_steps=10 ** 9, tie_break="k_eff")
    ref, _ = prune_path(AD, S0, cfg, label="x")
    ck = "/tmp/_fm_test_path.pt"
    if os.path.exists(ck):
        os.remove(ck)
    df = None
    n_calls = 0
    while df is None:                                   # deadline 0 s: exactly one step per call
        df = prune_path_resumable(AD, S0, cfg, "x", ck, deadline=time.time() + (0 if n_calls else 1))
        n_calls += 1
    cols = ["k_eff", "kind", "dk", "test_loss", "stop_mean", "stop_pass"]
    assert n_calls > 2
    assert ref[cols].equals(df[cols]) or (ref[cols].astype(str) == df[cols].astype(str)).all().all()
