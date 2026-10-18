"""Faithfulness of the Berman et al. (2026) baseline: our fdist_* schemes must
zero exactly the entries their own pruner classes zero.

Needs their repository and its dependency nngeometry:
    git clone https://github.com/edhirst/Fdist_Pruning
    pip install nngeometry
    FDIST_REPO=/path/to/Fdist_Pruning python -m pytest -q tests/test_fdist.py
Skipped otherwise.
"""
import copy
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nnprune import mlp as M  # noqa: E402
from nnprune.core import PruneConfig  # noqa: E402
from nnprune import fdist_baseline as FD  # noqa: E402

REPO = os.environ.get("FDIST_REPO")
if not REPO or not os.path.isdir(REPO):
    pytest.skip("set FDIST_REPO to a clone of edhirst/Fdist_Pruning", allow_module_level=True)
pytest.importorskip("nngeometry")
sys.path.insert(0, REPO)
from src.pruning import (FDistGlobalPruner, FDistIterativePruner,  # noqa: E402
                         FDistOneShotPruner, FDistPruner)


@pytest.fixture(scope="module")
def setup():
    data, _ = M.make_moons_data(n=600, seed=3)
    ad = M.MLPAdapter(data, widths=(2, 8, 8, 1))
    s, _ = M.train_mlp(ad, epochs=300, seed=3, verbose=False)
    return ad, s, FD.fisher_loader(ad, subset_size=200)


def _their_zeros(module):
    return {n: (p.detach() == 0) for n, p in module.named_parameters()}


def _our_zeros(state, names):
    return {n: ~state.masks[n] for n in names}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _ours(ad, s0, scheme, fracs, loader):
    names = FD._eligible(ad, s0, ())
    total = sum(int(v.numel()) for v in s0.params.values())
    running, pruned = s0.clone(), 0
    for f in fracs:
        target = int(round(f * total))
        if scheme == "fdist_one_shot":
            sq = FD._sqrtF_at(ad, s0, loader)
            running = s0.clone()
            FD._prune_lowest(running, {k: sq[k] * s0.params[k].abs() for k in names}, names, target)
            continue
        sq = {"fdist_iterative": lambda: FD._sqrtF_at(ad, running, loader),
              "fdist_global": lambda: FD._sqrtF_global(ad, running, loader, 3),
              "fdist_exact": lambda: FD._sqrtF_exact(ad, running, loader, 3)}[scheme]()
        pruned += FD._prune_lowest(running, {k: sq[k] * running.params[k].abs() for k in names},
                                   names, target - pruned)
    return _our_zeros(running, names)


def test_one_shot(setup):
    ad, s0, L = setup
    mod = FD.AdapterModule(ad, s0)
    FDistOneShotPruner({"pruning_threshold": 0.3, "fim_calculate_method": "backprop",
                        "prunable_exclude": []}).apply_pruning(mod, L)
    assert _same(_their_zeros(mod), _ours(ad, s0, "fdist_one_shot", [0.3], L))


@pytest.mark.parametrize("scheme,cls,extra", [
    ("fdist_iterative", FDistIterativePruner, {}),
    ("fdist_global", FDistGlobalPruner, {"f_dist_avg_points": 3}),
    ("fdist_exact", FDistPruner, {"f_dist_avg_points": 3, "f_dist_fast": True, "f_dist_workers": 1}),
])
def test_running_schemes(setup, scheme, cls, extra):
    ad, s0, L = setup
    mod = FD.AdapterModule(ad, s0)
    pr = cls({"pruning_step": 0.2, "fim_calculate_method": "backprop", "prunable_exclude": [], **extra})
    for _ in range(2):
        pr.apply_pruning(mod, L)
    assert _same(_their_zeros(mod), _ours(ad, s0, scheme, [0.2, 0.4], L))


def test_one_shot_transformer():
    from nnprune import transformer as T
    data = T.make_modadd_data(p=7, train_frac=0.5, seed=0)
    ad = T.TransformerAdapter(data, p=7, d=8, n_heads=2, d_head=2, d_mlp=16)
    s0 = ad.make_state(ad.init_params(seed=0))
    g = torch.Generator().manual_seed(1)
    for k in ("b_in", "b_out", "b_U"):          # zero-initialised: make them nonzero
        s0.params[k] = 0.1 * torch.randn(s0.params[k].shape, generator=g)
    L = FD.fisher_loader(ad, subset_size=20)
    mod = FD.AdapterModule(ad, s0)
    FDistOneShotPruner({"pruning_threshold": 0.3, "fim_calculate_method": "backprop",
                        "prunable_exclude": []}).apply_pruning(mod, L)
    names = FD._eligible(ad, s0, ())
    total = sum(int(s0.params[k].numel()) for k in names)
    sq = FD._sqrtF_at(ad, s0, L)
    ours = s0.clone()
    FD._prune_lowest(ours, {k: sq[k] * s0.params[k].abs() for k in names}, names, round(0.3 * total))
    assert _same(_their_zeros(mod), _our_zeros(ours, names))
