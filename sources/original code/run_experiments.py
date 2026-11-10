"""Command-line driver.  Results are cached as CSV under results/.

    python run_experiments.py mlp --seeds 123 7 42
    python run_experiments.py mlp-spectrum --seeds 123
    python run_experiments.py transformer --seeds 0
    python run_experiments.py fdist-mlp            # Berman et al. (2026) baselines on the cached MLPs
    python run_experiments.py fdist-transformer    # ... on the cached transformer
    python run_experiments.py fair-mlp             # crossed design: granularity x selection data
    python run_experiments.py fair-transformer
Add --recompute to ignore cached results, --threads N to set torch threads.
"""
import argparse

import torch

from nnprune import experiments as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["mlp", "mlp-spectrum", "transformer", "fdist-mlp", "fdist-transformer",
                                       "fair-mlp", "fair-transformer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.what == "mlp":
        out = E.run_mlp(a.outdir or "results/mlp", seeds=tuple(a.seeds or (123,)),
                        recompute=a.recompute, verbose=a.verbose)
        print(out["summary"].to_string())
    elif a.what == "mlp-spectrum":
        for s in (a.seeds or (123,)):
            r = E.mlp_spectrum(a.outdir or "results/mlp", seed=s, recompute=a.recompute)
            print(s, r["regime"], r["null_dim"], r["reason"])
    elif a.what.startswith("fair-"):
        out = E.run_fair(a.what.split("-", 1)[1], a.outdir, seeds=tuple(a.seeds) if a.seeds else None,
                         recompute=a.recompute, verbose=a.verbose)
        print(out["summary"].to_string())
    elif a.what.startswith("fdist-"):
        kind = a.what.split("-", 1)[1]
        out = E.run_fdist(kind, a.outdir, seeds=tuple(a.seeds) if a.seeds else None,
                          recompute=a.recompute, verbose=a.verbose)
        print(out["fdist_summary_loss"].to_string())
        print(out["fdist_summary_error"].to_string())
    else:
        out = E.run_transformer(a.outdir or "results/transformer", seeds=tuple(a.seeds or (0,)),
                                recompute=a.recompute, verbose=a.verbose)
        print(out["summary_loss"].to_string())
        print(out["summary_error"].to_string())
