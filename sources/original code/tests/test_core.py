"""Correctness checks: folds are exact, limits are the limits, k_eff counts the
cascade, the selection rule does what it says, the gap rule matches the original.

Run from the NNpruning folder:   python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nnprune.core import Move, PruneConfig, paired_stats, select_move, zero_entries  # noqa: E402
from nnprune import mlp as M  # noqa: E402
from nnprune import transformer as T  # noqa: E402
from nnprune import spectrum as S  # noqa: E402


@pytest.fixture(scope="module")
def mlp():
    data, _ = M.make_moons_data(n=400, seed=1)
    ad = M.MLPAdapter(data, widths=(2, 12, 12, 12, 1))
    s = ad.make_state(ad.init_params(seed=1))
    return ad, s


def _out(ad, s, split="test"):
    return ad.logits(ad.effective(s), s.struct, ad.data[split][0])


# ---------------------------------------------------------------- MLP -------

def test_fanin_zero_is_folded_exactly(mlp):
    ad, s0 = mlp
    s = s0.clone()
    zero_entries(s, "W1", torch.arange(12) + 3 * 12)          # unit 3 of layer 1: no fan-in
    before = _out(ad, s)
    ad.canonicalize(s)                                         # -> constant sigmoid(b), folded
    assert int(s.struct["type1"][3]) == M.REMOVED
    assert torch.allclose(before, _out(ad, s), atol=1e-6)


def test_fanout_zero_kills_fanin(mlp):
    ad, s0 = mlp
    s = s0.clone()
    col = torch.zeros_like(s.masks["W2"]); col[:, 5] = True
    zero_entries(s, "W2", col)
    before = _out(ad, s)
    k_before = ad.k_eff(s)
    ad.canonicalize(s)
    assert torch.allclose(before, _out(ad, s), atol=1e-7)
    assert k_before - ad.k_eff(s) == 12 + 1                    # fan-in row + bias freed


def test_unit_sat_equals_limit_and_k_count(mlp):
    ad, s0 = mlp
    for c, big in ((1.0, 60.0), (0.0, -60.0)):
        s = ad.apply(s0, Move("unit_sat", "h1", 1, {"l": 1, "i": 4, "c": c}))
        ref = s0.clone(); ref.params["b1"][4] = big           # b -> +-inf, numerically saturated
        assert torch.allclose(_out(ad, s), _out(ad, ref), atol=1e-5)
        assert ad.k_eff(s0) - ad.k_eff(s) == 12 + 1 + 12       # fan-in + bias + fan-out


def test_single_weight_to_inf_reaches_same_endpoint(mlp):
    """Layer >= 1 inputs are sigmoids (> 0): one incoming weight -> +inf saturates
    the whole unit, the same endpoint as unit_sat c=1."""
    ad, s0 = mlp
    s = ad.apply(s0, Move("unit_sat", "h1", 1, {"l": 1, "i": 4, "c": 1.0}))
    ref = s0.clone(); ref.params["W1"][4, 7] = 1e5
    assert torch.allclose(_out(ad, s), _out(ad, ref), atol=1e-4)


def test_step_equals_heaviside_limit(mlp):
    ad, s0 = mlp
    s = ad.apply(s0, Move("unit_step", "h0", 0, {"l": 0, "i": 2, "j": 1, "s": -1.0}))
    ref = s0.clone(); ref.params["W0"][2, 1] = -1e6
    X = ad.data["test"][0]
    keep = X[:, 1].abs() > 1e-2
    a = ad.logits(ad.effective(s), s.struct, X[keep])
    b = ad.logits(ad.effective(ref), ref.struct, X[keep])
    assert torch.allclose(a, b, atol=1e-5)
    assert int(s.struct["type0"][2]) == M.STEP


def test_merge_of_exact_duplicate_is_exact(mlp):
    ad, s0 = mlp
    s = s0.clone()
    s.params["W1"][6] = s.params["W1"][2]; s.params["b1"][6] = s.params["b1"][2]
    before = _out(ad, s)
    s2 = ad.apply(s, Move("unit_merge", "h1", 1, {"l": 1, "i": 6, "i2": 2, "alpha": 1.0, "beta": 0.0}))
    assert torch.allclose(before, _out(ad, s2), atol=1e-6)


# ---------------------------------------------------------- selection -------

def test_paired_stats():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m, se = paired_stats(a, a)
    assert m == 0 and se == 0
    m, se = paired_stats(a + torch.tensor([0.1, -0.1, 0.1, -0.1]), a)
    assert abs(m) < 1e-12 and se > 0


def test_select_move_tie_break():
    base = torch.zeros(400)
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(400, generator=g) * 0.1

    def r(dk, layer, shift, n=noise):
        c = base + shift + n
        return {"crit": c, "mean_crit": float(c.mean()), "mean_loss": float(c.mean()),
                "dk": dk, "layer": layer, "admissible": True}
    cands = [r(1, 0, 0.0, torch.zeros(400)),      # best, removes 1
             r(50, 2, 0.001),                     # tied (noisy), removes 50, deep layer
             r(50, 0, 0.001),                     # tied, removes 50, shallow layer
             r(500, 1, 1.0)]                      # clearly worse: not in the tie set
    ch, n_adm, n_ties = select_move(cands, PruneConfig(tie_break="k_eff_then_layer"))
    assert n_ties == 3 and ch["dk"] == 50 and ch["layer"] == 0
    ch, _, _ = select_move(cands, PruneConfig(tie_break="layer"))
    assert ch["layer"] == 0
    ch, _, _ = select_move(cands, PruneConfig(tie_break="none"))
    assert ch["dk"] == 1


# -------------------------------------------------------- transformer -------

@pytest.fixture(scope="module")
def tf():
    data = T.make_modadd_data(p=11, train_frac=0.5, seed=0)
    ad = T.TransformerAdapter(data, p=11, d=8, n_heads=2, d_head=2, d_mlp=16)
    s = ad.make_state(ad.init_params(seed=0))
    return ad, s


def test_dead_value_kills_head_exactly(tf):
    ad, s0 = tf
    s = s0.clone()
    m = torch.zeros_like(s.masks["W_V"]); m[1] = True
    zero_entries(s, "W_V", m)
    before = _out(ad, s)
    ad.canonicalize(s)
    assert not bool(s.struct["head_alive"][1])
    assert torch.allclose(before, _out(ad, s), atol=1e-6)


def test_relu_fanin_fold_exact(tf):
    ad, s0 = tf
    s = s0.clone()
    s.params["b_in"][3] = 0.7
    zero_entries(s, "W_in", torch.arange(ad.d) + 3 * ad.d)
    before = _out(ad, s)
    ad.canonicalize(s)
    assert int(s.struct["unit_type"][3]) == M.REMOVED
    assert torch.allclose(before, _out(ad, s), atol=1e-6)


def test_weighted_lowrank_full_rank_reproduces():
    g = torch.Generator().manual_seed(0)
    A0, B0 = torch.randn(8, 3, generator=g), torch.randn(8, 3, generator=g)
    X = torch.randn(50, 8, generator=g)
    A, B = T.weighted_lowrank(A0 @ B0.T, X.T @ X, torch.eye(8), 3)
    assert torch.allclose(A @ B.T, A0 @ B0.T, atol=1e-4)


def test_attn_uniform_is_s_to_zero_limit(tf):
    ad, s0 = tf
    s = ad.apply(s0, Move("attn_uniform", "head0", 1, {"h": 0}))
    ref = s0.clone(); ref.params["W_Q"][0] *= 1e-7
    assert torch.allclose(_out(ad, s), _out(ad, ref), atol=1e-5)


def test_qk_null_matches_prediction(tf):
    """Constant query token: the QK block's null dimension is 2 d d_h - r_K."""
    ad, s0 = tf
    X = torch.cat([ad.data[k][0] for k in ("train", "val", "test")])
    pred = S.transformer_predicted_nulls(ad, s0, X)
    r = S.classify_block(ad, s0, S.head_blocks(ad, s0, 0)["QK"], X, R=4)
    assert r["regime"] == "structural" and r["null_dim"] == pred["QK"]


# ------------------------------------------------------------ spectrum -------

def test_classify_spectrum_matches_criteria_bench():
    root = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, root)
    try:
        from criteria_bench.spectrum import classify_spectrum as orig
    except Exception:
        pytest.skip("criteria_bench not importable here")
    rng = np.random.default_rng(0)
    for _ in range(20):
        ev = np.concatenate([10 ** rng.uniform(-8, 0, 20), 10 ** rng.uniform(-300, -14, rng.integers(0, 4))])
        a, b = S.classify_spectrum(ev), orig(ev)
        assert (a.regime, a.null_dim) == (b.regime, b.null_dim)


def test_noise_floor_removes_sign_artefact():
    """A sloppy spectrum decaying into rounding noise of both signs: without the
    floor the rule counts negative rounding errors; with it, it refuses."""
    ev = np.concatenate([np.logspace(0, -18, 60), -np.abs(np.logspace(-17, -19, 20))])
    assert S.classify_spectrum(ev).regime == "structural"
    assert S.classify_spectrum(ev, noise_floor="auto").regime == "under_determined"
