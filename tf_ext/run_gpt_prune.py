"""
Her pruning procedure on the toy GPT (gpt_adapter.py), with and without the stream move.

Arms (her TF_BASE candidate budgets; her FAIR_COMMON rule: valA chooses, valB stops,
tie-break fewest parameters then loss, admissibility off):
  S | valA                       her structural moves
  S+stream | valA                + stream_mean (exact LayerNorm route)
  stream only | valA
  W+S | valA | magnitude         + single-weight chunks of 64, magnitude proposer
  W+S+stream | valA | magnitude
  S+stream+new | valA            + the new moves (gpt_adapter.NEW_KINDS)
  W+S+stream+new | valA | magnitude
  new only | valA
  S+allstream+new | valA         + stream moves at every granularity (stream_site, stream_read)
  W+S+allstream+new | valA | magnitude
  allstream only | valA
Stage 1 (exact folds) is applied first; frac_removed_orig counts from the trained model.
Margins: the anchored valB test  mean dL <= 2 SE + rho * base val loss  is recorded with
rho = 5% (tol_abs), and each run ends 50 steps after its first failure there (or at 97%
removed); stop points for smaller rho are recomputed afterwards from stop_mean, stop_se.

Usage:  L=2 SEED=0 THREADS=2 python3 run_gpt_prune.py ARM [ARM ...]
        writes results_gpt_prune/L<L>_seed<k>__<arm>.csv
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpt_adapter as GA  # noqa: E402
import gpt_toy as G  # noqa: E402
from nnprune.core import PruneConfig, prune_path  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
OUT = os.environ.get("GPT_PRUNE_RESULTS", os.path.join(HERE, "results_gpt_prune"))
L, SEED = int(os.environ.get("L", "2")), int(os.environ.get("SEED", "0"))
RHO_STOP = 0.05
COMMON = dict(weight_chunk=64, weight_score="magnitude", n_struct=24, n_merge=16, screen_n=256, refresh_every=20,
              tie_break="k_eff", criterion="loss", select_split="valA", stop_split="valB", stop_mode="mark",
              select_admissibility=False, post_stop_steps=50, max_frac=0.97)
ARMS = {
    "S | valA": dict(weight_per_site=0, struct_kinds=GA.HER_KINDS),
    "S+stream | valA": dict(weight_per_site=0, struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS),
    "stream only | valA": dict(weight_per_site=0, struct_kinds=GA.STREAM_KINDS),
    "W+S | valA | magnitude": dict(weight_per_site=3, struct_kinds=GA.HER_KINDS),
    "W+S+stream | valA | magnitude": dict(weight_per_site=3, struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS),
    # the new moves (emb/pos/unemb rank cuts, token merge, head merge) with everything else
    "S+stream+new | valA": dict(weight_per_site=0, struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS + GA.NEW_KINDS),
    "W+S+stream+new | valA | magnitude": dict(weight_per_site=3,
                                              struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS + GA.NEW_KINDS),
    "new only | valA": dict(weight_per_site=0, struct_kinds=GA.NEW_KINDS),
    # stream moves at every granularity (global, per LayerNorm site, per reader matrix)
    "S+allstream+new | valA": dict(weight_per_site=0, struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS
                                   + GA.STREAM_FINE_KINDS + GA.NEW_KINDS),
    "W+S+allstream+new | valA | magnitude": dict(weight_per_site=3, struct_kinds=GA.HER_KINDS + GA.STREAM_KINDS
                                                 + GA.STREAM_FINE_KINDS + GA.NEW_KINDS),
    "allstream only | valA": dict(weight_per_site=0, struct_kinds=GA.STREAM_KINDS + GA.STREAM_FINE_KINDS),
}


def fname(arm):
    return os.path.join(OUT, f"L{L}_seed{SEED}__" + arm.replace(" | ", "__").replace(" ", "_").replace("+", "p") + ".csv")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ad, s0, base = GA.load(L=L, seed=SEED)
    p_orig = G.n_params(base["params"])
    z0 = G.forward(base["params"], ad.cfg, ad.data["val"][0][:512])
    err = float((ad.logits(ad.effective(s0), s0.struct, ad.data["val"][0][:512]) - z0).abs().max())
    base_val = ad.metrics(s0, "val")["loss"]
    print(f"L={L} seed {SEED}: p = {p_orig}, after Stage 1 k = {ad.k_eff(s0)} "
          f"({p_orig - ad.k_eff(s0)} removed exactly, max |logit change| {err:.1e}); val loss {base_val:.4f}", flush=True)
    for arm in sys.argv[1:]:
        if os.path.exists(fname(arm)):
            print(f"SKIP {arm} (exists)", flush=True)
            continue
        t = time.time()
        cfg = PruneConfig(**{**COMMON, **ARMS[arm], "tol_abs": RHO_STOP * base_val, "seed": SEED,
                             "verbose": True, "log_every": 10})
        df, _ = prune_path(ad, s0, cfg, label=f"L{L} s{SEED} {arm}")
        df["seed"], df["L"], df["method"] = SEED, L, arm
        df["p_orig"], df["stage1_removed"], df["stage1_max_logit_err"] = p_orig, p_orig - ad.k_eff(s0), err
        df["frac_removed_orig"] = 1 - df["k_eff"] / p_orig
        df.to_csv(fname(arm), index=False)
        print(f"DONE L={L} seed {SEED} {arm}: {len(df) - 1} steps, {100 * df['frac_removed_orig'].iloc[-1]:.1f}% "
              f"removed, {time.time() - t:.0f}s", flush=True)
    print(f"ALL DONE L={L} seed {SEED}", flush=True)
