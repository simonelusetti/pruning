"""
Low-rank conv move on Fashion-MNIST, same protocol, data, trained networks and
one-shot baselines as run_fm.py (the networks and baseline curves are reused from
results_fm, so the arms are compared on identical starting points).

Arms:
  S-cnn | valA              control: the channel moves of run_fm (rerun with the new adapter)
  rank-only | valA          only conv_lowrank / conv_rankcut (plus cascades)
  S-cnn+rank | valA         channel moves + rank moves
  W+S-cnn+rank | valA | obd everything, plus single weights

Usage (repeat until "0 units left"):
  CNN_RESULTS=results_fm_rank python run_fm_rank.py [budget_seconds] [seeds...]
"""
import os
import shutil
import sys
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn_fm import CNN_KINDS, load_fashion  # noqa: E402
from cnn_fm_rank import RANK_KINDS, CNNAdapterFMRank  # noqa: E402
from nnprune.core import PruneConfig  # noqa: E402
from resumable import prune_path_resumable  # noqa: E402
from run_fm import BASE, COMMON, STATICS  # noqa: E402

torch.set_num_threads(4)
OUT = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results_fm_rank"))
SRC = os.path.join(HERE, "results_fm")
UNITS = os.path.join(OUT, "units")
VARIANTS = {
    "S-cnn | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=CNN_KINDS),
    "rank-only | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=RANK_KINDS),
    "S-cnn+rank | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=CNN_KINDS + RANK_KINDS),
    "W+S-cnn+rank | valA | obd": dict(select_split="valA", weight_score="obd", struct_kinds=CNN_KINDS + RANK_KINDS),
}


def ufile(seed, name, root=UNITS):
    return os.path.join(root, f"seed{seed}__{name.replace(' | ', '__').replace(' ', '_')}.csv")


if __name__ == "__main__":
    os.makedirs(UNITS, exist_ok=True)
    t0 = time.time()
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    seeds = [int(a) for a in sys.argv[2:]] or [0, 1]
    for seed in seeds:                                   # reuse the one-shot baselines of results_fm
        for n in STATICS:
            if not os.path.exists(ufile(seed, n)):
                shutil.copy(ufile(seed, n, os.path.join(SRC, "units")), ufile(seed, n))
    stop = False
    for seed in seeds:
        todo = [n for n in VARIANTS if not os.path.exists(ufile(seed, n))]
        if not todo:
            continue
        data, _ = load_fashion(seed, n_valA=2000, n_prune_train=2000)
        ad = CNNAdapterFMRank(data, stats_n=1000)
        s0 = ad.upgrade(torch.load(os.path.join(SRC, f"base_state_seed{seed}.pt"), weights_only=False))
        for name in todo:
            if time.time() - t0 > budget:
                stop = True
                break
            t = time.time()
            cfg = PruneConfig(**{**BASE, **COMMON, **VARIANTS[name], "seed": seed, "verbose": False})
            df = prune_path_resumable(ad, s0, cfg, name, ufile(seed, name) + ".ckpt", t0 + budget)
            if df is None:
                print(f"[seed {seed}] {name}: checkpointed ({time.time() - t:.0f}s)", flush=True)
                stop = True
                break
            df["seed"] = seed
            df.to_csv(ufile(seed, name), index=False)
            print(f"[seed {seed}] {name}: done ({time.time() - t:.0f}s)", flush=True)
        if stop:
            break
    left = [(s, n) for s in seeds for n in VARIANTS if not os.path.exists(ufile(s, n))]
    print(f"{len(left)} units left", flush=True)
    if not left:
        P = pd.concat([pd.read_csv(ufile(s, n)) for s in seeds for n in VARIANTS], ignore_index=True)
        S = pd.concat([pd.read_csv(ufile(s, n)) for s in seeds for n in STATICS], ignore_index=True)
        P.to_csv(os.path.join(OUT, "paths.csv"), index=False)
        S.to_csv(os.path.join(OUT, "statics.csv"), index=False)
        print("merged into paths.csv and statics.csv")
