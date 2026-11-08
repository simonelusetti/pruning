from __future__ import annotations

import random
import torch
from omegaconf import OmegaConf
from pruning.engine import run_path
from pruning.results import Recorder


def validate(cfg):
    supported = {"local": {"classic", "berman", "sparsegpt", "discrete"},
                 "mnist": {"classic", "discrete", "berman"},
                 "opt": {"classic", "continuous", "sparsegpt", "wanda", "flap"},
                 "published": {"sparsegpt", "wanda", "flap"}}
    family, method = cfg.benchmark.family, cfg.branch.name
    if family not in supported or method not in supported[family]:
        raise ValueError(f"unsupported benchmark/method: {cfg.benchmark.name}/{method}")
    if family == "local" and ((method == "discrete" and cfg.benchmark.name != "moons") or
                               (method == "sparsegpt" and cfg.benchmark.name != "modadd")):
        raise ValueError(f"{method} is not maintained for {cfg.benchmark.name}")
    p, s = cfg.protocol, cfg.sampling
    if p.get("trace", False) and (not p.selection_check or family != "opt" or
                                  method != "continuous" or cfg.branch.options.get("deletion_unit") != "hidden_neuron"):
        raise ValueError("proposal tracing requires neuron continuous with validation enabled")
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
    if schedule.mode not in {"greedy", "progressive", "independent"}:
        raise ValueError("unknown pruning schedule")
    if (method in {"classic", "discrete"}) != (schedule.mode == "greedy"):
        raise ValueError("classic and discrete need greedy schedules; other methods need sparsity targets")
    if schedule.mode != "greedy":
        from pruning.engine import targets
        if min(schedule.get("spacing", .05), schedule.get("fine_spacing", .05),
               schedule.get("coarse_spacing", .05)) <= 0:
            raise ValueError("target spacing must be positive")
        grid = targets(schedule, schedule.get("maximum", p.max_removed))
        if not grid or any(not 0 < value < 1 for value in grid) or grid != sorted(set(grid)):
            raise ValueError("sparsity targets must be increasing, distinct, and between zero and one")


def run_experiment(cfg, run):
    validate(cfg)
    torch.set_num_threads(cfg.runtime.threads)
    torch.manual_seed(cfg.benchmark.seed)
    random.seed(cfg.benchmark.seed)
    family, name = cfg.benchmark.family, cfg.branch.name
    if family == "published":
        from pruning.baselines.native import run_native
        return run_native(cfg, run)
    options = OmegaConf.merge(
        OmegaConf.to_container(cfg.benchmark.get("methods", {}).get(name, OmegaConf.create({})), resolve=True),
        OmegaConf.to_container(cfg.branch.options, resolve=True))
    if family == "opt":
        from pruning.models.opt import OptProblem
        problem = OptProblem(cfg, options)
    elif family == "mnist":
        from pruning.models.mnist import MnistProblem
        problem = MnistProblem(cfg)
    else:
        from pruning.models.local import LocalProblem
        problem = LocalProblem(cfg)
    if name == "berman":
        from pruning.baselines.berman import Berman
        method = Berman(options)
    elif family == "local" and name == "sparsegpt":
        from pruning.baselines.modadd import ModaddSparseGPT
        method = ModaddSparseGPT(options)
    elif family == "local":
        from pruning.methods.local import LocalGreedy
        method = LocalGreedy(cfg, options)
    elif name == "continuous":
        from pruning.methods.continuous import Continuous
        method = Continuous(problem, options)
    elif name == "classic":
        from pruning.methods.classic import OptClassic, MnistClassic
        method = (OptClassic if family == "opt" else MnistClassic)(options, cfg.protocol)
    elif name == "discrete":
        from pruning.methods.discrete import MnistDiscrete
        method = MnistDiscrete(options, cfg.protocol)
    else:
        from pruning.baselines.language import LanguageBaseline
        method = LanguageBaseline(name, options)
    metadata = {"benchmark": cfg.benchmark.name, "method": name, "device": cfg.runtime.device,
                "schedule": OmegaConf.to_container(cfg.branch.schedule, resolve=True),
                "protocol": OmegaConf.to_container(cfg.protocol, resolve=True),
                "method_options": OmegaConf.to_container(options, resolve=True)}
    if family == "opt":
        s = cfg.sampling
        metadata.update(model=cfg.benchmark.model, model_dtype=cfg.benchmark.dtype,
                        sampling=OmegaConf.to_container(s, resolve=True),
                        parameter_accounting="decoder_linear_weights_and_biases",
                        pruning_modules=list(problem.names), accounting_modules=list(problem.accounting_names),
                        deletion_unit=options.get("deletion_unit"), calibration={
                            "dataset": s.dataset, "nsamples": s.calibration_sequences,
                            "seqlen": s.context, "seed": s.seed, "refresh": s.refresh,
                            "b_sequences": s.b_sequences if name == "continuous" else None,
                            "b_context": s.b_context if name == "continuous" else None,
                            "b_tokens": problem.settings.b_tokens if name == "continuous" else None})
    elif family == "local":
        from pruning.models.local import checkpoint_sha256
        metadata["checkpoint_sha256"] = checkpoint_sha256(cfg.benchmark.name)
    return run_path(problem, method, cfg.protocol, cfg.branch.schedule, Recorder(run, metadata))
