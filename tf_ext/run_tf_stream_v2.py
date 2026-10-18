"""
Round-4 transformer comparison with the stream move, version 2:
  * stream candidates are 'fresh' slots (recomputed from the current network when
    applied; StreamAdapter(fresh=True)), 6 pooled + 2 per depth;
  * seeds 0, 1, 2.  Seed 0 is her cached base model; seeds 1 and 2 are trained here
    with her own recipe (transformer.train_transformer, 3000 steps, data split = seed),
    exactly as nnprune.experiments.run_transformer does, and saved under OUT.

Arms (TF_BASE settings, valA chooses, valB stops):
  S | valA,  W+S | valA | fdist                     her moves only
  S+stream | valA,  W+S+stream | valA | fdist       + stream_mean
  stream only | valA

Usage:  SEED=1 python run_tf_stream_v2.py ARM [ARM ...]     (writes OUT/seed<k>__<arm>.csv)
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
from nnprune.experiments import TF_BASE, split_validation  # noqa: E402
from tf_stream import STREAM_KINDS, StreamAdapter  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
OUT = os.environ.get("TF_STREAM_RESULTS", os.path.join(HERE, "results_tf_stream_v2"))
SEED = int(os.environ.get("SEED", "0"))
HER_KINDS = ("head_off", "head_mean", "attn_uniform", "attn_fixed", "qk_rank", "ov_rank",
             "unit_sat", "unit_mean", "unit_merge")
ARMS = {
    "S | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=HER_KINDS),
    "W+S | valA | fdist": dict(select_split="valA", weight_score="fdist", struct_kinds=HER_KINDS),
    "S+stream | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=HER_KINDS + STREAM_KINDS),
    "W+S+stream | valA | fdist": dict(select_split="valA", weight_score="fdist",
                                      struct_kinds=HER_KINDS + STREAM_KINDS),
    "stream only | valA": dict(select_split="valA", weight_per_site=0, struct_kinds=STREAM_KINDS),
}
LEGACY_COMMON = dict(tie_break="k_eff", criterion="loss", stop_split="valB",
                     select_admissibility=False)


def fname(arm, seed):
    return os.path.join(OUT, f"seed{seed}__" + arm.replace(" | ", "__").replace(" ", "_").replace("+", "p") + ".csv")


def base_state(seed, data):
    if seed == 0:
        return torch.load(os.path.join(os.path.dirname(HERE), "checkpoints",
                                       "modadd_seed0.pt"), weights_only=False)
    f = os.path.join(OUT, f"base_state_seed{seed}.pt")
    if not os.path.exists(f):
        her = T.TransformerAdapter(data, p=59)
        t = time.time()
        s0, _ = T.train_transformer(her, steps=3000, seed=seed, log_every=250, verbose=False)
        torch.save(s0, f)
        m = {sp: her.metrics(s0, sp) for sp in ("train", "val", "test")}
        print(f"TRAINED seed {seed} in {time.time() - t:.0f}s: " +
              "  ".join(f"{k} loss {v['loss']:.2e} acc {v['acc']:.3f}" for k, v in m.items()), flush=True)
    return torch.load(f, weights_only=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = T.make_modadd_data(p=59, train_frac=0.5, seed=SEED)
    s_her = base_state(SEED, data)
    data = split_validation(data, SEED)
    ad = StreamAdapter(data, p=59, n_pool=6, n_each=2, fresh=True)
    s0 = ad.upgrade(s_her)
    for arm in sys.argv[1:]:
        if os.path.exists(fname(arm, SEED)):
            print(f"SKIP seed {SEED} {arm} (exists)", flush=True)
            continue
        t = time.time()
        cfg = PruneConfig(**{**TF_BASE, **LEGACY_COMMON, **ARMS[arm],
                             "seed": SEED, "verbose": True, "log_every": 50})
        try:
            df, _ = prune_path(ad, s0, cfg, label=arm)
        except Exception as e:                      # noqa: BLE001
            print(f"FAILED seed {SEED} {arm}: {type(e).__name__}: {e}", flush=True)
            raise
        df["seed"] = SEED
        df["method"] = arm
        df.to_csv(fname(arm, SEED), index=False)
        print(f"DONE seed {SEED} {arm}: {len(df) - 1} steps, {100 * df['frac_removed'].iloc[-1]:.1f}% removed, "
              f"{time.time() - t:.0f}s", flush=True)
    print(f"ALL DONE seed {SEED}", flush=True)
