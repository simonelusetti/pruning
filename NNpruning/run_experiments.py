import argparse
from pathlib import Path

import torch

from nnprune.experiments import checkpoint_sha256, run_benchmark, save_run_artifact
from nnprune.fisher import PRESETS, resolve_strategy, validate_strategy


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
    parser.add_argument("--fisher-preset", choices=PRESETS)
    parser.add_argument("--fisher-geometry", choices=("full", "axb", "axi"))
    parser.add_argument("--fisher-deletion", choices=("miqp", "obs"))
    parser.add_argument("--fisher-compensation",
                        choices=("sparse_miqp", "dense", "sparsegpt"))
    parser.add_argument("--fisher-update-scope", choices=("global", "layer"))
    parser.add_argument("--fisher-structure", choices=("reward", "post", "none"))
    parser.add_argument("--fisher-decision", choices=("validate", "proxy"))
    parser.add_argument("--fisher-lambda-u", type=float, default=1e-8)
    parser.add_argument("--fisher-lambda-d", type=float, default=1e-7)
    parser.add_argument("--fisher-mu", type=float, default=1e-4)
    parser.add_argument("--fisher-pool", type=int, default=8)
    parser.add_argument("--fisher-time-limit", type=float, default=60.0)
    parser.add_argument("--fisher-gap", type=float, default=0.01)
    parser.add_argument("--fisher-bound", type=float, default=10.0)
    parser.add_argument("--fisher-log", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fisher_requested = any(move in {"fisher", "fisher_general"}
                           for move in args.moves)
    fisher_overrides = (args.fisher_geometry, args.fisher_deletion,
                        args.fisher_compensation, args.fisher_update_scope,
                        args.fisher_structure, args.fisher_decision)
    if (args.fisher_preset or any(value is not None for value in fisher_overrides)) \
            and not fisher_requested:
        parser.error("Fisher strategy options require --moves fisher")
    if fisher_requested and args.benchmark != "moons":
        parser.error("the Fisher MIQP backend currently supports two-moons only")
    if fisher_requested and args.weights:
        parser.error("Fisher strategies already select scalar deletions")
    preset = args.fisher_preset or ("fisher_general" if fisher_requested else None)
    strategy = (resolve_strategy(
        preset, geometry=args.fisher_geometry, deletion=args.fisher_deletion,
        compensation=args.fisher_compensation, structure=args.fisher_structure,
        update_scope=args.fisher_update_scope,
        decision=args.fisher_decision) if preset else None)
    if strategy is not None:
        try:
            strategy = validate_strategy(strategy, "mlp")
        except ValueError as error:
            parser.error(str(error))
    path, _ = run_benchmark(
        args.benchmark, groups=args.moves, device=args.device,
        max_loss_error=args.max_loss_error, verbose=not args.quiet,
        weights=args.weights,
        fisher_strategy=strategy,
        fisher_options={
            "lambda_u": args.fisher_lambda_u, "lambda_d": args.fisher_lambda_d,
            "mu": args.fisher_mu, "pool": args.fisher_pool,
            "time_limit": args.fisher_time_limit, "gap": args.fisher_gap,
            "bound": args.fisher_bound, "log": args.fisher_log},
        screen_n=args.screen_n, supervisor_protocol=args.supervisor_protocol,
        validation_split_seed=args.validation_split_seed)
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
        fisher=path.attrs["fisher"])
    print(f"removed={row.frac_removed:.3%} val_loss={row.val_loss:.6f} "
          f"test_loss={row.test_loss:.6f}")
    print("moves=" + ", ".join(f"{kind}:{count}" for kind, count in counts.items()))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
