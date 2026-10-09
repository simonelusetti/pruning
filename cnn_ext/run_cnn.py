"""
Round-4 protocol of Veronica's draft (her `run_fair`), on the digits CNN.

Arms (her core.prune_path, with FAIR_COMMON: moves chosen on valA (or train),
anchored paired 2-sigma stop test on valB, tie-break fewest remaining parameters,
log loss, paths continued past the stop point by at most 300 steps, up to 97% removed):
  W | valA | obd          single weights only (her OBD proposer)
  W | valA | magnitude    single weights only (magnitude proposer)
  S-naive | valA          her MLP moves ported as they are (input-side folds)
  S-cnn | valA            + output-side constant folds and the spatial merge
  W+S-naive | valA | obd
  W+S-cnn | valA | obd
  W+S-cnn | train | obd   same, moves chosen on training data (held-out-in-loop control)
One-shot baselines, cut by the same valB test: magnitude, obd, random, mean ablation.

Usage:  python run_cnn.py [seeds...]      (default: 0 1 2)
Writes results/paths.csv, results/statics.csv (appends per seed, skips cached seeds).
"""
import os
import sys
import time

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn import CNN_KINDS, NAIVE_KINDS, CNNAdapter, make_digits_data, train_cnn  # noqa: E402
from nnprune.core import PruneConfig, baseline_path, prune_path, unit_baseline_path  # noqa: E402
from nnprune.experiments import FAIR_COMMON, split_validation  # noqa: E402

torch.set_num_threads(4)
BASE = dict(weight_chunk=1, weight_per_site=8, n_struct=32, n_merge=16, refresh_every=10)
VARIANTS = {
    "W | valA | obd": dict(select_split="valA", weight_score="obd", use_struct=False),
    "W | valA | magnitude": dict(select_split="valA", weight_score="magnitude", use_struct=False),
    "S-naive | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=NAIVE_KINDS),
    "S-cnn | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=CNN_KINDS),
    "W+S-naive | valA | obd": dict(select_split="valA", weight_score="obd", struct_kinds=NAIVE_KINDS),
    "W+S-cnn | valA | obd": dict(select_split="valA", weight_score="obd", struct_kinds=CNN_KINDS),
    "W+S-cnn | train | obd": dict(select_split="train", weight_score="obd", struct_kinds=CNN_KINDS),
}
OUT = os.environ.get("CNN_RESULTS", os.path.join(HERE, "results"))


def run_seed(seed):
    data = make_digits_data(seed)
    ad = CNNAdapter(split_validation(data, seed))
    s0 = train_cnn(ad, seed=seed)
    torch.save(s0, os.path.join(OUT, f"base_state_seed{seed}.pt"))
    m = {sp: ad.metrics(s0, sp) for sp in ("train", "valA", "valB", "test")}
    print(f"[seed {seed}] base: " + "  ".join(f"{k} loss {v['loss']:.4f} acc {v['acc']:.3f}" for k, v in m.items()), flush=True)
    paths, statics = [], []
    for name, v in VARIANTS.items():
        cfg = PruneConfig(**{**BASE, **FAIR_COMMON, **v, "post_stop_steps": 300, "seed": seed, "verbose": False})
        t = time.time()
        df, _ = prune_path(ad, s0, cfg, label=name)
        df["seed"] = seed
        paths.append(df)
        print(f"[seed {seed}] {name}: {len(df) - 1} steps to {100 * df['frac_removed'].iloc[-1]:.1f}%, "
              f"{time.time() - t:.0f}s", flush=True)
    cfg = PruneConfig(**{**BASE, **FAIR_COMMON, "seed": seed, "verbose": False})
    for b in ("magnitude", "obd", "random"):
        d = baseline_path(ad, s0, b, cfg)
        d["seed"] = seed
        statics.append(d)
    d = unit_baseline_path(ad, s0, cfg)
    d["seed"] = seed
    statics.append(d)
    return pd.concat(paths, ignore_index=True), pd.concat(statics, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    pp, sp = os.path.join(OUT, "paths.csv"), os.path.join(OUT, "statics.csv")
    P = pd.read_csv(pp) if os.path.exists(pp) else pd.DataFrame()
    S = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    for seed in seeds:
        if len(P) and seed in set(P["seed"]):
            print(f"[seed {seed}] cached")
            continue
        p, s = run_seed(seed)
        P = pd.concat([P, p], ignore_index=True)
        S = pd.concat([S, s], ignore_index=True)
        P.to_csv(pp, index=False)
        S.to_csv(sp, index=False)
