from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .plans import build_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "NNpruning"))


def _numeric_row(row: dict) -> dict[str, float]:
    return {key: float(value) for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def _rows(result: dict) -> list[dict]:
    return result.get("history", result.get("curve", result.get("runs", [])))


def _record_progress(run, artifact: Path, seen: int) -> int:
    if not artifact.exists():
        return seen
    try:
        rows = _rows(json.loads(artifact.read_text()))
    except (OSError, json.JSONDecodeError):
        return seen
    for index in range(seen, len(rows)):
        if isinstance(rows[index], dict):
            values = _numeric_row(rows[index])
            if values:
                run.push_log(values, step=index)
    return len(rows)


def _local(cfg: DictConfig, run) -> dict[str, float]:
    from nnprune.experiments import checkpoint_sha256, run_benchmark, save_run_artifact
    from nnprune.fisher import resolve_strategy, validate_strategy

    benchmark = cfg.benchmark.name
    branch = cfg.branch.name
    if benchmark == "modadd" and branch not in {"classic", "classic_weights"}:
        raise ValueError(f"{branch} is not a maintained modular-addition branch")
    if branch not in {"classic", "classic_weights", "fisher_general"}:
        raise ValueError(f"{branch} is not a local pruning branch")
    if cfg.benchmark.supervisor_protocol and (
            benchmark != "moons" or branch not in {"classic", "classic_weights"} or
            cfg.benchmark.max_loss_error != .01 or
            cfg.benchmark.validation_split_seed != 123):
        raise ValueError("supervisor protocol requires moons Classic, tolerance .01, split seed 123")
    fisher = branch == "fisher_general"
    options = OmegaConf.to_container(cfg.branch.options, resolve=True)
    if not fisher and options:
        raise ValueError(f"{branch} has no configurable branch options on {benchmark}")
    weights = branch == "classic_weights"
    strategy = validate_strategy(resolve_strategy("fisher_general"), "mlp") if fisher else None
    groups = ("fisher_general",) if fisher else ("classic",)
    path, _ = run_benchmark(
        benchmark, groups=groups, device=cfg.runtime.device,
        max_loss_error=float(cfg.benchmark.max_loss_error), verbose=False,
        weights=weights, fisher_strategy=strategy,
        fisher_options=options if fisher else None,
        validation_split_seed=int(cfg.benchmark.validation_split_seed),
        supervisor_protocol=bool(cfg.benchmark.supervisor_protocol),
    )
    artifact = run.path / "result.json"
    save_run_artifact(
        artifact, path, benchmark=benchmark, move_groups=groups,
        checkpoint_sha256=checkpoint_sha256(benchmark),
        validation_split_seed=path.attrs["validation_split_seed"],
        device=cfg.runtime.device, max_loss_error=path.attrs["tol_frac"],
        weights=weights, n_sigma=2.0, fisher=path.attrs["fisher"],
    )
    result = json.loads(artifact.read_text())
    for index, row in enumerate(result["history"]):
        run.push_log(_numeric_row(row), step=index)
    row = result["summary"]
    return {"removed_percent": 100 * float(row["frac_removed"]),
            "validation_loss": float(row["val_loss"]),
            "test_loss": float(row["test_loss"]),
            "stop_step": float(result["stop_index"])}


def execute(cfg: DictConfig, run) -> dict[str, float]:
    torch.set_num_threads(int(cfg.runtime.threads))
    benchmark, branch = cfg.benchmark.name, cfg.branch.name
    if benchmark in {"moons", "modadd"} and branch in {
            "classic", "classic_weights", "fisher_general"}:
        return _local(cfg, run)
    plan = build_plan(cfg, run.path)
    tracked = run.path / plan.artifact if plan.artifact else None
    process = subprocess.Popen([sys.executable, "-m", "pruning_experiments.worker",
                                plan.module, *plan.arguments], cwd=ROOT)
    seen = 0
    try:
        while process.poll() is None:
            if tracked is not None:
                seen = _record_progress(run, tracked, seen)
            time.sleep(1)
        if tracked is not None:
            _record_progress(run, tracked, seen)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    artifacts = ([run.path / plan.artifact] if plan.artifact else
                 list(run.path.rglob("command.json")))
    if len(artifacts) != 1 or not artifacts[0].exists():
        raise RuntimeError(f"expected one result artifact for {benchmark}/{branch}: {artifacts}")
    result = json.loads(artifacts[0].read_text())
    if result.get("status") not in (None, "complete"):
        raise RuntimeError(f"experiment did not complete: {result.get('status')}")
    history = _rows(result)
    endpoint = result.get("stopping_point") or (history[-1] if history else {})
    metrics = _numeric_row(endpoint)
    for key in ("stop_index", "seconds", "target_sparsity"):
        value = result.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            metrics[key] = float(value)
    if not metrics:
        metrics = {"completed": 1.0}
    return metrics
