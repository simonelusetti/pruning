"""Plan or execute baselines in their authors' native released settings."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from native_baselines.specs import SPECS

HERE = Path(__file__).resolve().parents[1] / "native_baselines"


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()


def verify_repo(method: str, repo: Path) -> str:
    spec = SPECS[method]
    if not (repo / spec["script"]).is_file():
        raise FileNotFoundError(f"missing {repo / spec['script']}; run bootstrap.py first")
    head = git_head(repo)
    if head != spec["commit"]:
        raise RuntimeError(f"{method} source is {head}, expected {spec['commit']}")
    return head


def build_command(method: str, python: str, repo: Path, model: str, sparsity: float,
                  seed: int, nsamples: int, cache_dir: Path, output: Path, args) -> list[str]:
    spec = SPECS[method]
    script = str(repo / spec["script"])
    if method == "sparsegpt":
        command = [python, script, model, args.dataset, "--seed", str(seed),
                   "--nsamples", str(nsamples), "--percdamp", str(args.percdamp),
                   "--blocksize", str(args.blocksize), "--sparsity", str(sparsity)]
        if args.save_model:
            command += ["--save", str(output / "model")]
        return command
    if method == "wanda":
        command = [python, script, "--model", model, "--seed", str(seed),
                   "--nsamples", str(nsamples), "--sparsity_ratio", str(sparsity),
                   "--sparsity_type", args.sparsity_type, "--prune_method", "wanda",
                   "--cache_dir", str(cache_dir), "--save", str(output / "official_results")]
        if args.zero_shot:
            command.append("--eval_zero_shot")
        if args.save_model:
            command += ["--save_model", str(output / "model")]
        return command
    command = [python, script, "--model", model, "--seed", str(seed),
               "--nsamples", str(nsamples), "--pruning_ratio", str(sparsity),
               "--remove_heads", "-1", "--metrics", "WIFV", "--structure", "AL-AM",
               "--prune_method", "flap", "--cache_dir", str(cache_dir), "--eval"]
    if args.flap_mask_only:
        command.append("--unstr")
    if args.save_model:
        command += ["--save_model", str(output / "model")]
    return command


def package_versions(python: str):
    code = ("import importlib.metadata as m,json; p=['torch','transformers','datasets','accelerate','sentencepiece']; "
            "out={}; "
            "exec(\"for x in p:\\n try: out[x]=m.version(x)\\n except m.PackageNotFoundError: out[x]=None\"); "
            "print(json.dumps(out))")
    try:
        result = subprocess.run([python, "-c", code], check=True, text=True,
                                capture_output=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def has_cuda(python: str) -> bool:
    result = subprocess.run([python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"])
    return result.returncode == 0


def version_mismatches(method: str, versions):
    if versions is None:
        return ["could not inspect packages"]
    return [f"{name}={versions.get(name)!r} (expected {expected})"
            for name, expected in SPECS[method]["versions"].items()
            if versions.get(name) is None or not versions[name].startswith(expected)]


def run_and_tee(command, cwd, env, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def parser_for(method):
    spec = SPECS[method]
    parser = argparse.ArgumentParser(description=spec["paper"])
    parser.add_argument("method", choices=tuple(SPECS))
    parser.add_argument("--execute", action="store_true",
                        help="execute; without this flag only print and record the plan")
    parser.add_argument("--repo", type=Path, default=HERE / "vendor" / method)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default=spec["model"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=spec["nsamples"])
    parser.add_argument("--sparsity", type=float, default=spec["sparsity"])
    parser.add_argument("--include-dense", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help="execute even when packages differ from the official environment")
    parser.add_argument("--dataset", choices=("c4", "wikitext2", "ptb"),
                        default=spec.get("dataset", "c4"))
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--sparsity-type", choices=("unstructured", "4:8", "2:4"),
                        default="unstructured")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--flap-mask-only", action="store_true",
                        help="use FLAP's --unstr masking path instead of physical compression")
    return parser


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] not in SPECS:
        raise SystemExit(f"usage: native {{{','.join(SPECS)}}} [options]")
    method = arguments[0]
    args = parser_for(method).parse_args(arguments)
    head = verify_repo(method, args.repo)
    if not 0 <= args.sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if method == "wanda" and args.sparsity_type != "unstructured" and args.sparsity != 0.5:
        raise ValueError("official Wanda N:M modes require --sparsity 0.5")

    settings = ([0.0] if args.include_dense else []) + [args.sparsity]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HF_HOME"] = str(args.cache_dir)
    env["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")
    env["TRANSFORMERS_CACHE"] = str(args.cache_dir / "transformers")
    versions = package_versions(args.python)
    mismatches = version_mismatches(method, versions)
    for sparsity in settings:
        tag = "dense" if sparsity == 0 else f"s{sparsity:g}".replace(".", "p")
        if method == "wanda":
            tag += f"_{args.sparsity_type.replace(':', '-') }"
        output = args.output_root / method / Path(args.model).name / f"seed{args.seed}_{tag}"
        output.mkdir(parents=True, exist_ok=True)
        command = build_command(method, args.python, args.repo, args.model, sparsity,
                                args.seed, args.nsamples, args.cache_dir, output, args)
        record = {
            "method": method, "paper": SPECS[method]["paper"], "source_url": SPECS[method]["url"],
            "source_commit": head, "model": args.model, "calibration_dataset": args.dataset,
            "seed": args.seed, "nsamples": args.nsamples, "target_sparsity": sparsity,
            "gpu": args.gpu, "command": command, "cwd": str(args.repo),
            "package_versions": versions, "expected_versions": SPECS[method]["versions"],
            "version_mismatches": mismatches, "status": "planned",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        record_path = output / "command.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(" ".join(command), flush=True)
        if not args.execute:
            continue
        if mismatches and not args.allow_version_mismatch:
            raise RuntimeError("native environment mismatch: " + "; ".join(mismatches))
        if not has_cuda(args.python):
            raise RuntimeError(f"{args.python} cannot see CUDA; native official programs require a GPU")
        record["status"] = "running"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        code = run_and_tee(command, args.repo, env, output / "stdout.log")
        record["status"] = "complete" if code == 0 else "failed"
        record["returncode"] = code
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if code:
            raise SystemExit(code)
