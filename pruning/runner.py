from __future__ import annotations

import random
import torch
from omegaconf import OmegaConf
from pruning.engine import run_path, targets
from pruning.results import Recorder

OURS = {"joint", "joint_simple", "joint_simple_global", "joint_submodular", "singular"}
SUPPORTED = {"local": {"classic", "berman", "sparsegpt", "joint", "reference"},
             "mnist": {"classic", "joint", "berman", "reference"} | OURS,
             "network": {"reference"} | OURS,
             "opt": {"classic", "sparsegpt", "wanda", "flap", "reference", "native"} | OURS,
             "llama": {"reference", "native"} | OURS}


def label(branch):
    """Method name in plots: our methods and the reference controls are named by their settings."""
    if branch.name in OURS:
        parts = [branch.name, branch.scope, branch.unit, "affine" if branch.affine else "linear", branch.curvature]
        parts += ["replay"] if branch.replay else []
        parts += [branch.objective] if branch.objective != "kl" else []
        parts += ["uncompensated"] if branch.compensation == "none" else []
        if branch.name == "joint" and (branch.options.update_pool != branch.options.pool or branch.options.lambda_u == 0):
            parts += [f"{branch.options.update_pool} updates" + (" free" if branch.options.lambda_u == 0 else "")]
        return " · ".join(parts)
    if branch.name == "reference":
        return f"reference · {branch.options.score} · {branch.scope}"
    return branch.get("label", branch.name)


def validate(cfg):
    family, method = cfg.benchmark.family, cfg.branch.name
    if family not in SUPPORTED or method not in SUPPORTED[family]:
        raise ValueError(f"unsupported benchmark/method: {cfg.benchmark.name}/{method}")
    if family == "local" and method == "sparsegpt" and cfg.benchmark.name != "modadd":
        raise ValueError(f"{method} is not maintained for {cfg.benchmark.name}")
    p, s = cfg.protocol, cfg.sampling
    if p.max_steps < 1 or not 0 < p.max_removed <= 1 or p.rho < 0 or p.n_sigma < 0:
        raise ValueError("invalid pruning limits")
    if p.candidate_selection not in {"loss", "proxy"} or p.stop_mode not in {"halt", "mark"}:
        raise ValueError("invalid candidate selection or stopping mode")
    if p.reporting_every < 0 or s.refresh not in {"step", "fixed"}:
        raise ValueError("invalid reporting cadence or calibration refresh")
    if min(s.calibration_sequences, s.b_sequences) < 1 or s.b_tokens < 0:
        raise ValueError("calibration counts must be positive and B token cap nonnegative")
    if min(s.context, s.b_context, s.selection_context, s.heldout_context) < 2:
        raise ValueError("sequence contexts must contain at least two tokens")
    if s.heldout_sequences < 2 or (p.selection_check and s.selection_sequences < 2):
        raise ValueError("reporting and enabled selection checks need at least two sequences")
    if method == "classic" and p.candidate_selection == "loss" and s.candidate_sequences < 2:
        raise ValueError("loss-based candidate selection needs at least two sequences")
    if min(s.evaluation_batch_size, s.b_batch_size, s.activation_batch_size) < 1:
        raise ValueError("evaluation, activation, and B batch sizes must be positive")
    schedule = cfg.branch.schedule
    greedy = {"classic", "joint", "joint_simple_global"}
    if schedule.mode not in {"greedy", "progressive", "independent"} or (schedule.mode == "greedy") != (method in greedy):
        raise ValueError("classic, joint and joint_simple_global are greedy; every other method follows targets")
    if method == "berman" and cfg.branch.options.get("scheme", "iterative") in {"one_shot", "magnitude", "fim"} and schedule.mode != "independent":
        raise ValueError("Berman's static schemes require an independent schedule")
    if method == "reference" and cfg.branch.options.refresh == "step" and schedule.mode != "progressive":
        raise ValueError("Refreshed ranked baselines require a progressive schedule")
    if schedule.mode != "greedy":
        grid = targets(schedule, schedule.get("maximum", p.max_removed))
        if not grid or any(not 0 < value < 1 for value in grid) or grid != sorted(set(grid)):
            raise ValueError("sparsity targets must be increasing, distinct, and between zero and one")


def problem_for(cfg, options):
    family = cfg.benchmark.family
    if family == "network":
        from pruning.models.network import NetworkProblem
        return NetworkProblem(cfg)
    if family in {"opt", "llama"}:
        from pruning.models.opt import OptProblem
        return OptProblem(cfg, options)
    if family == "mnist":
        from pruning.models.mnist import MnistProblem
        return MnistProblem(cfg)
    from pruning.models.local import LocalProblem
    return LocalProblem(cfg)


def method_for(cfg, problem, options):
    name, family = cfg.branch.name, cfg.benchmark.family
    branch = OmegaConf.merge(OmegaConf.to_container(cfg.branch, resolve=True), {"options": options})
    if name in {"joint_simple", "joint_simple_global"}:
        from pruning.methods.joint_simple import JointSimple
        return JointSimple(problem, branch)
    if name == "joint_submodular":
        from pruning.methods.joint_submodular import JointSubmodular
        return JointSubmodular(problem, branch)
    if name == "singular":
        from pruning.methods.singular import Singular
        return Singular(problem, branch)
    if name == "joint":
        from pruning.methods.joint import Joint
        return Joint(problem, branch)
    if name == "reference":
        from pruning.baselines.reference import ReferenceBaseline
        return ReferenceBaseline(problem, branch)
    if name == "berman":
        from pruning.baselines.berman import Berman
        return Berman(options)
    if family == "local" and name == "sparsegpt":
        from pruning.baselines.modadd import ModaddSparseGPT
        return ModaddSparseGPT(options)
    if family == "local":
        from pruning.methods.local import LocalGreedy
        return LocalGreedy(cfg, options)
    if name == "classic":
        from pruning.methods.classic import OptClassic, MnistClassic
        return (OptClassic if family == "opt" else MnistClassic)(options, cfg.protocol)
    from pruning.baselines.language import LanguageBaseline
    return LanguageBaseline(name, options)


def run_experiment(cfg, run):
    validate(cfg)
    torch.set_num_threads(cfg.runtime.threads)
    torch.manual_seed(cfg.benchmark.seed)
    random.seed(cfg.benchmark.seed)
    metadata = {"benchmark": cfg.benchmark.name, "method": cfg.branch.name, "label": label(cfg.branch),
                "device": cfg.runtime.device, "branch": OmegaConf.to_container(cfg.branch, resolve=True),
                "protocol": OmegaConf.to_container(cfg.protocol, resolve=True)}
    if cfg.branch.name == "native":
        from pruning.baselines.native import NativeMethod, NativeProblem
        problem = NativeProblem(cfg, run)
        recorder = Recorder(run, {**metadata, **problem.metadata})
        return run_path(problem, NativeMethod(problem), cfg.protocol, cfg.branch.schedule, recorder)
    options = OmegaConf.merge(
        OmegaConf.to_container(cfg.benchmark.get("methods", {}).get(cfg.branch.name, OmegaConf.create({})), resolve=True),
        OmegaConf.to_container(cfg.branch.options, resolve=True))
    problem = problem_for(cfg, options)
    method = method_for(cfg, problem, options)
    if cfg.benchmark.family in {"opt", "llama"}:
        metadata.update(model=cfg.benchmark.model, model_dtype=cfg.benchmark.dtype,
                        sampling=OmegaConf.to_container(cfg.sampling, resolve=True),
                        pruning_modules=list(problem.names), accounting_modules=list(problem.accounting_names))
    elif cfg.benchmark.family == "local":
        from pruning.models.local import checkpoint_sha256
        metadata["checkpoint_sha256"] = checkpoint_sha256(cfg.benchmark.name)
    return run_path(problem, method, cfg.protocol, cfg.branch.schedule, Recorder(run, metadata))
