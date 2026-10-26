from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .core import PruneConfig, prune_path
from .fisher import FisherStrategy
from . import mlp as M
from . import transformer as T

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINT_NAMES = {"moons": "moons_seed123.pt", "modadd": "modadd_seed0.pt"}

MLP_BASE = dict(weight_chunk=1, weight_per_site=8, n_struct=32, n_merge=16,
                refresh_every=10)
TF_BASE = dict(weight_chunk=64, weight_per_site=3, n_struct=24, n_merge=16,
               screen_n=512, refresh_every=20)


def split_validation(data: dict, seed: int = 0) -> dict:
    out = dict(data)
    X, y = data["val"]
    if len(X) < 2:
        raise ValueError("validation data needs at least two examples")
    generator = torch.Generator(device=X.device).manual_seed(seed + 1000)
    order = torch.randperm(len(X), generator=generator, device=X.device)
    half = len(X) // 2
    out["valA"] = X[order[:half]], y[order[:half]]
    out["valB"] = X[order[half:]], y[order[half:]]
    return out


def load_benchmark(name: str, device="cpu"):
    device = torch.device(device)
    if name == "moons":
        data, _ = M.make_moons_data(seed=123)
        adapter = M.MLPAdapter(data).to(device)
        checkpoint = CHECKPOINTS / CHECKPOINT_NAMES[name]
        train = lambda: M.train_mlp(adapter, seed=123, verbose=False)[0]
    elif name == "modadd":
        data = T.make_modadd_data(seed=0)
        adapter = T.TransformerAdapter(data, p=59).to(device)
        checkpoint = CHECKPOINTS / CHECKPOINT_NAMES[name]
        train = lambda: T.train_transformer(adapter, steps=3000, seed=0,
                                             verbose=False)[0]
    else:
        raise ValueError(f"unknown benchmark: {name}")
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False).to(device)
    else:
        state = train()
        CHECKPOINTS.mkdir(exist_ok=True)
        torch.save(state.to("cpu"), checkpoint)
    return adapter, state


def checkpoint_sha256(name: str) -> str:
    path = CHECKPOINTS / CHECKPOINT_NAMES[name]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(name: str, groups=("classic",), device="cpu",
                  max_loss_error=0.05, verbose=True, weights=False,
                  fisher_strategy: FisherStrategy | None = None,
                  fisher_options=None, screen_n=None,
                  supervisor_protocol=False, validation_split_seed=None):
    adapter, state = load_benchmark(name, device)
    validation_seed = (123 if supervisor_protocol else 0) if validation_split_seed is None \
        else validation_split_seed
    adapter.data = split_validation(adapter.data, validation_seed)
    base = dict({"moons": MLP_BASE, "modadd": TF_BASE}[name])
    if not weights:
        base["weight_per_site"] = 0
    if screen_n is not None:
        base["screen_n"] = screen_n
    protocol = {}
    if supervisor_protocol:
        if name != "moons" or tuple(groups) != ("classic",):
            raise ValueError("the supervisor protocol is defined for classic two-moons")
        base["screen_n"] = None
        max_loss_error = 0.01
        protocol = dict(tie_break="k_eff", stop_mode="mark",
                        select_admissibility=False, post_stop_steps=10 ** 9,
                        max_frac=0.97, seed=123)
        if weights:
            protocol["weight_score"] = "fdist"
    options = {f"fisher_{key}": value
               for key, value in (fisher_options or {}).items()}
    cfg = PruneConfig(**base, **protocol, **options, struct_kinds=tuple(groups),
                      fisher_strategy=fisher_strategy,
                      select_split="valA", stop_split="valB", n_sigma=2.0,
                      tol_frac=max_loss_error, verbose=verbose)
    path, final = prune_path(adapter, state, cfg, label="+".join(groups))
    path.attrs["candidate_config"] = {
        key: getattr(cfg, key) for key in (
            "weight_per_site", "n_struct", "n_merge", "refresh_every",
            "screen_n", "n_mc", "expand_on_fail")}
    path.attrs["fisher"] = {
        "strategy": (cfg.fisher_strategy.as_dict()
                     if cfg.fisher_strategy is not None else None),
        **{key: getattr(cfg, f"fisher_{key}") for key in
        ("lambda_u", "lambda_d", "mu", "pool", "time_limit", "gap",
         "bound", "log")}}
    path.attrs["supervisor_protocol"] = supervisor_protocol
    path.attrs["validation_split_seed"] = validation_seed
    return path, final


def save_run_artifact(output, path, **metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(right >= left for left, right in
           zip(path["k_eff"], path["k_eff"].iloc[1:])):
        raise ValueError("pruning history must strictly reduce k_eff")
    for column in ("frac_removed", "train_loss", "val_loss", "test_loss"):
        if not all(math.isfinite(float(value)) for value in path[column]):
            raise ValueError(f"non-finite {column} in pruning history")
    if "within_stop" in path:
        flags = path["within_stop"].astype(bool).tolist()
        if flags != sorted(flags, reverse=True):
            raise ValueError("within_stop must be a true prefix")
        if path.attrs.get("decision", "validate") == "validate":
            tolerance = (path.attrs["tol_abs"] + path.attrs.get("tol_frac", 0.0)
                         * path.attrs["stop_reference"])
            for _, row in path.iloc[1:].iterrows():
                passed = row.stop_mean <= path.attrs["n_sigma"] * row.stop_se + tolerance
                if bool(row.stop_pass) != bool(passed):
                    raise ValueError("stored stopping decision does not match its statistics")
    history = json.loads(path.to_json(orient="records"))
    accepted = path.loc[path["within_stop"]] if "within_stop" in path else path
    selected = accepted.loc[accepted["kind"] != ""]
    selection_columns = ["step", "kind", "site", "layer", "move", "dk", "pred_cost"]
    selection_columns += [column for column in (
        "n_deleted", "n_updates", "closure_savings",
        "search_objective", "solver_objective", "solver_gap",
        "solver_seconds", "compensation_bound_active") if column in selected]
    artifact = {
        **metadata,
        "result_schema": 2,
        "status": "complete",
        "protocol": dict(path.attrs),
        "stop_index": int(accepted.index[-1]),
        "summary": json.loads(accepted.iloc[-1].to_json()),
        "move_counts": {str(k): int(v) for k, v in selected["kind"].value_counts().items()},
        "selections": json.loads(selected[selection_columns].to_json(orient="records")),
        "history": history,
    }
    if "within_stop" in path and bool((~path["within_stop"]).any()):
        artifact["rejected_candidate"] = json.loads(
            path.loc[~path["within_stop"]].iloc[0].to_json())
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n")
    os.replace(temporary, output)
