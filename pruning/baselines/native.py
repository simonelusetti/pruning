"""Pinned external baselines; only these programs need separate interpreters."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf
from pruning.baselines.specs import SPECS
from pruning.baselines.result_artifacts import write_json
from pruning.paths import BASELINE_CACHE, BASELINE_VENDOR

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
                  seed: int, nsamples: int, cache_dir: Path, output: Path, options) -> list[str]:
    spec = SPECS[method]
    script = str(repo / spec["script"])
    if method == "sparsegpt":
        command = [python, script, model, options.dataset, "--seed", str(seed),
                   "--nsamples", str(nsamples), "--percdamp", str(options.percdamp),
                   "--blocksize", str(options.blocksize), "--sparsity", str(sparsity)]
        if options.save_model:
            command += ["--save", str(output / "model")]
        return command
    if method == "wanda":
        command = [python, script, "--model", model, "--seed", str(seed),
                   "--nsamples", str(nsamples), "--sparsity_ratio", str(sparsity),
                   "--sparsity_type", options.sparsity_type, "--prune_method", "wanda",
                   "--cache_dir", str(cache_dir), "--save", str(output / "official_results")]
        if options.zero_shot:
            command.append("--eval_zero_shot")
        if options.save_model:
            command += ["--save_model", str(output / "model")]
        return command
    command = [python, script, "--model", model, "--seed", str(seed),
               "--nsamples", str(nsamples), "--pruning_ratio", str(sparsity),
               "--remove_heads", "-1", "--metrics", "WIFV", "--structure", "AL-AM",
               "--prune_method", "flap", "--cache_dir", str(cache_dir), "--eval"]
    if options.flap_mask_only:
        command.append("--unstr")
    if options.save_model:
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



def run_native(cfg, run):
    method, benchmark, options = cfg.branch.name, cfg.benchmark, cfg.branch.options
    spec = SPECS[method]
    if (benchmark.source != method or benchmark.model != spec["model"] or
            benchmark.dataset != spec["dataset"] or benchmark.nsamples != spec["nsamples"]):
        raise ValueError("published benchmark must match its pinned model and data settings")
    if not str(cfg.runtime.device).startswith("cuda"):
        raise ValueError("published implementations require runtime.device=cuda:N")
    if not 0 <= options.sparsity < 1:
        raise ValueError("sparsity must be in [0, 1)")
    if method == "wanda" and options.sparsity_type != "unstructured" and options.sparsity != .5:
        raise ValueError("official Wanda N:M modes require sparsity 0.5")
    repo = BASELINE_VENDOR / method
    head = verify_repo(method, repo)
    python = cfg.runtime.python or sys.executable
    versions = package_versions(python)
    mismatches = version_mismatches(method, versions)
    if mismatches and not options.allow_version_mismatch:
        raise RuntimeError("native environment mismatch: " + "; ".join(mismatches))
    if not has_cuda(python):
        raise RuntimeError(f"{python} cannot see CUDA")
    arguments = OmegaConf.merge(OmegaConf.to_container(options, resolve=True), {"dataset": benchmark.dataset})
    command = build_command(method, python, repo, benchmark.model, options.sparsity,
                            benchmark.seed, benchmark.nsamples, BASELINE_CACHE, run.path, arguments)
    record = {"method": method, "benchmark": "published", "status": "running",
              "source_commit": head, "package_versions": versions, "version_mismatches": mismatches,
              "model": benchmark.model, "dataset": benchmark.dataset,
              "target_sparsity": options.sparsity, "command": command, "cwd": str(repo)}
    write_json(run.path / "result.json", record)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(cfg.runtime.device).partition(":")[2] or "0",
               HF_HOME=str(BASELINE_CACHE), HF_DATASETS_CACHE=str(BASELINE_CACHE / "datasets"),
               TRANSFORMERS_CACHE=str(BASELINE_CACHE / "transformers"))
    code = run_and_tee(command, repo, env, run.path / "stdout.log")
    record.update(status="complete" if code == 0 else "failed", returncode=code)
    write_json(run.path / "result.json", record)
    if code:
        raise RuntimeError(f"published {method} exited with status {code}")
    return {"target_sparsity": float(options.sparsity)}
