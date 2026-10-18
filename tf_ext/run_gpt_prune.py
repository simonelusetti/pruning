import argparse
import os
import sys
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpt_adapter as GA  # noqa: E402
import gpt_toy as G  # noqa: E402
from nnprune.core import PruneConfig, prune_path  # noqa: E402
from nnprune.experiments import save_run_artifact  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves", nargs="+", default=("classic",),
                        help="move groups or individual move names")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-loss-error", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    adapter, state, base = GA.load(device=args.device)
    cfg = PruneConfig(
        struct_kinds=tuple(args.moves),
        weight_chunk=64, weight_per_site=3 if args.weights else 0,
        weight_score="magnitude", n_struct=24, n_merge=16, screen_n=256,
        refresh_every=20, tie_break="k_eff", select_split="valA",
        stop_split="valB", n_sigma=2.0, tol_frac=args.max_loss_error,
        verbose=True, log_every=10,
    )
    path, _ = prune_path(adapter, state, cfg, label="+".join(args.moves))
    output = args.output or Path(HERE) / "results" / (
        f"gpt_{'-'.join(args.moves)}.json")
    save_run_artifact(
        output, path, benchmark="gpt", move_groups=args.moves,
        device=args.device, max_loss_error=args.max_loss_error,
        weights=args.weights, n_sigma=2.0)
    row = path.loc[path["within_stop"]].iloc[-1]
    original = G.n_params(base["params"])
    print(f"removed={1 - row.k_eff / original:.3%} val_loss={row.val_loss:.6f} "
          f"test_loss={row.test_loss:.6f}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
