import argparse
from pathlib import Path

import torch

from nnprune.experiments import checkpoint_sha256, run_benchmark, save_run_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("moons", "modadd"))
    parser.add_argument("--moves", nargs="+", default=("classic",),
                        help="move groups or individual move names")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-loss-error", type=float, default=0.05,
                        help="allowed relative increase over the original validation loss")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--screen-n", type=int)
    parser.add_argument("--validation-split-seed", type=int)
    parser.add_argument("--supervisor-protocol", action="store_true",
                        help="reproduce the original classic two-moons controlled protocol")
    parser.add_argument("--fisher-quant-gamma", type=float, default=0.2)
    parser.add_argument("--fisher-general-lambda-u", type=float, default=1e-8)
    parser.add_argument("--fisher-general-lambda-d", type=float, default=1e-7)
    parser.add_argument("--fisher-general-mu", type=float, default=1e-4)
    parser.add_argument("--fisher-general-window", type=int, default=16)
    parser.add_argument("--fisher-general-min-savings", type=int, default=1)
    parser.add_argument("--fisher-general-pool", type=int, default=8)
    parser.add_argument("--fisher-general-starts", type=int, default=2)
    parser.add_argument("--fisher-general-iters", type=int, default=4)
    parser.add_argument("--fisher-general-update-pool", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path, _ = run_benchmark(
        args.benchmark, args.moves, args.device, args.max_loss_error,
        not args.quiet, args.weights, args.fisher_quant_gamma, {
            "lambda_u": args.fisher_general_lambda_u,
            "lambda_d": args.fisher_general_lambda_d,
            "mu": args.fisher_general_mu,
            "window": args.fisher_general_window,
            "min_savings": args.fisher_general_min_savings,
            "pool": args.fisher_general_pool,
            "starts": args.fisher_general_starts,
            "iters": args.fisher_general_iters,
            "update_pool": args.fisher_general_update_pool},
        args.screen_n, args.supervisor_protocol, args.validation_split_seed)
    accepted = path.loc[path["within_stop"]] if "within_stop" in path else path
    row = accepted.iloc[-1]
    counts = accepted.loc[accepted["kind"] != "", "kind"].value_counts()
    output = args.output or Path(__file__).resolve().parent / "results" / (
        f"{args.benchmark}_{'-'.join(args.moves)}.json")
    save_run_artifact(
        output, path, benchmark=args.benchmark, move_groups=args.moves,
        checkpoint_sha256=checkpoint_sha256(args.benchmark),
        validation_split_seed=path.attrs["validation_split_seed"],
        device=args.device, max_loss_error=path.attrs["tol_frac"],
        weights=args.weights, n_sigma=2.0,
        fisher_quant_gamma=args.fisher_quant_gamma,
        fisher_general=path.attrs["fisher_general"])
    print(f"removed={row.frac_removed:.3%} val_loss={row.val_loss:.6f} "
          f"test_loss={row.test_loss:.6f}")
    print("moves=" + ", ".join(f"{kind}:{count}" for kind, count in counts.items()))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
