"""
Her round-4 transformer comparison with the stream-direction move added.

Same base model, data split (valA chooses, valB stops), TF_BASE and FAIR_COMMON
settings as nnprune.experiments.run_fair('transformer').  Arms:
  S | valA                      her structural moves only (reproduction check against
                                NNpruning/results/fair_transformer/paths.csv)
  S+stream | valA               + stream_mean
  stream only | valA            stream_mean alone
  W+S+stream | valA | fdist     + single weights with the Fisher-distance proposer
Her 'W+S | valA | fdist' path is read from her results, not rerun.

Usage:  python run_tf_stream.py ARM [ARM ...]      (each arm writes results_tf_stream/<arm>.csv)
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "NNpruning"))
from nnprune import transformer as T  # noqa: E402
from nnprune.core import PruneConfig, prune_path  # noqa: E402
from nnprune.experiments import FAIR_COMMON, TF_BASE, split_validation  # noqa: E402
from tf_stream import STREAM_KINDS, StreamAdapter  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
OUT = os.environ.get("TF_STREAM_RESULTS", os.path.join(HERE, "results_tf_stream"))
HER_KINDS = ("head_off", "head_mean", "attn_uniform", "attn_fixed", "qk_rank", "ov_rank",
             "unit_sat", "unit_mean", "unit_merge")
ARMS = {
    "S | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=HER_KINDS),
    "S+stream | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=HER_KINDS + STREAM_KINDS),
    "stream only | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=STREAM_KINDS),
    "W+S+stream | valA | fdist": dict(select_split="valA", weight_score="fdist",
                                      struct_kinds=HER_KINDS + STREAM_KINDS),
}


def fname(arm):
    return os.path.join(OUT, arm.replace(" | ", "__").replace(" ", "_").replace("+", "p") + ".csv")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = split_validation(T.make_modadd_data(p=59, train_frac=0.5, seed=0), 0)
    ad = StreamAdapter(data, p=59, n_pool=int(os.environ.get("N_POOL", "4")))
    s0 = ad.upgrade(torch.load(os.path.join(os.path.dirname(HERE), "NNpruning", "results", "transformer",
                                            "base_state_seed0.pt"), weights_only=False))
    for arm in sys.argv[1:]:
        if os.path.exists(fname(arm)):
            print(f"SKIP {arm} (exists)", flush=True)
            continue
        t = time.time()
        cfg = PruneConfig(**{**TF_BASE, **FAIR_COMMON, **ARMS[arm], "seed": 0, "verbose": True, "log_every": 25})
        try:
            df, _ = prune_path(ad, s0, cfg, label=arm)
        except Exception as e:                      # noqa: BLE001
            print(f"FAILED {arm}: {type(e).__name__}: {e}", flush=True)
            raise
        df["seed"] = 0
        df["method"] = arm
        df.to_csv(fname(arm), index=False)
        print(f"DONE {arm}: {len(df) - 1} steps, {100 * df['frac_removed'].iloc[-1]:.1f}% removed, "
              f"{time.time() - t:.0f}s", flush=True)
    print("ALL DONE", flush=True)
