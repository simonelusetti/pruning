"""Run pruning baselines on the pre-LN toy GPT.

All methods start from Stage 1, so ``frac_removed_orig`` also credits the exact
LayerNorm/attention folds. Each curve uses the same anchored valB rule and the
accepted endpoint is printed without writing result files.

Examples
--------
  L=2 SEED=0 python3 run_gpt_baselines.py classical
  L=2 SEED=0 python3 run_gpt_baselines.py modern
  L=2 SEED=0 python3 run_gpt_baselines.py fdist
  L=2 SEED=0 python3 run_gpt_baselines.py fdist --include-exact

``fdist_exact`` is opt-in: on this 110k-parameter GPT it requires forward-mode
probes for every live coordinate at every sparsity point and is not practically
comparable in runtime to the other baselines.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "NNpruning"))

import gpt_adapter as GA  # noqa: E402
import gpt_toy as G  # noqa: E402
from nnprune.core import PruneConfig, baseline_path, unit_baseline_path  # noqa: E402
from nnprune.fdist_baseline import fdist_path  # noqa: E402
from modern_baselines import modern_baseline_path  # noqa: E402

CLASSICAL = ("magnitude", "obd", "random", "mean_ablation")
MODERN = ("sparsegpt", "wanda", "flap")
FDIST = ("fdist_one_shot", "fdist_iterative", "fdist_global")
DEFAULT_FRACS = np.round(np.concatenate([np.arange(0, 0.5, 0.02),
                                         np.arange(0.5, 0.96, 0.05)]), 4)


def annotate(df, ad, s0, base, method):
    p_orig = G.n_params(base["params"])
    out = df.copy()
    out["method"] = method
    out["L"] = ad.L
    out["p_orig"] = p_orig
    out["stage1_removed"] = p_orig - ad.k_eff(s0)
    out["frac_removed_orig"] = 1.0 - out["k_eff"] / p_orig
    return out


def run_one(ad, s0, base, method, cfg, fracs, calib_n=None, percdamp=0.01, blocksize=128):
    if method == "mean_ablation":
        df = unit_baseline_path(ad, s0, cfg, label=method)
    elif method in MODERN:
        df = modern_baseline_path(ad, s0, method, cfg, fracs=fracs,
                                  calib_n=calib_n, percdamp=percdamp,
                                  blocksize=blocksize)
    elif method.startswith("fdist_"):
        df = fdist_path(ad, s0, method, cfg, fracs=fracs, exclude=("W_pos",),
                        active_total=True, verbose=cfg.verbose)
    else:
        df = baseline_path(ad, s0, method, cfg, fracs=fracs)
    return annotate(df, ad, s0, base, method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=("classical", "modern", "fdist", "all"))
    ap.add_argument("--include-exact", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--calib-n", type=int, default=None,
                    help="override official calibration counts (128 for Wanda/SparseGPT, 2048 for FLAP)")
    ap.add_argument("--percdamp", type=float, default=0.01,
                    help="SparseGPT Hessian damping (official default: 0.01)")
    ap.add_argument("--blocksize", type=int, default=128,
                    help="SparseGPT reconstruction block size (official default: 128)")
    ap.add_argument("--fracs", type=float, nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", choices=CLASSICAL + MODERN + FDIST + ("fdist_exact",),
                    help="run only these methods (useful for resuming or smoke tests)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    L, seed = int(os.environ.get("L", "2")), int(os.environ.get("SEED", "0"))
    ad, s0, base = GA.load(L=L, seed=seed)
    cfg = PruneConfig(seed=seed, verbose=args.verbose, criterion="loss",
                      select_split="valA", stop_split="valB",
                      tol_frac=0.05)
    fracs = np.asarray(args.fracs if args.fracs is not None else DEFAULT_FRACS)

    methods = []
    if args.family in ("classical", "all"):
        methods.extend(CLASSICAL)
    if args.family in ("modern", "all"):
        methods.extend(MODERN)
    if args.family in ("fdist", "all"):
        methods.extend(FDIST)
        if args.include_exact:
            methods.append("fdist_exact")
    if args.methods is not None:
        methods = list(args.methods)

    for method in methods:
        t0 = time.time()
        df = run_one(ad, s0, base, method, cfg, fracs, args.calib_n,
                     args.percdamp, args.blocksize)
        col = "within_tolerance_loss"
        accepted = df[df[col]]
        stop = accepted.iloc[-1] if len(accepted) else df.iloc[0]
        print(f"DONE {method}: {100 * stop.frac_removed_orig:.1f}% of original parameters "
              f"at the 5% margin, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
