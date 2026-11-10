from __future__ import annotations

import random
import torch
from omegaconf import OmegaConf
from pruning.engine import run_path
from pruning.results import Recorder


def validate(cfg):
    supported = {"local": {"classic", "berman", "sparsegpt", "discrete"},
                 "mnist": {"classic", "discrete", "discrete_simple", "berman"},
                 "network": {"discrete", "discrete_simple"},
                 "opt": {"classic", "continuous", "canonical_sparsegpt", "discrete_simple", "sparsegpt", "wanda", "flap"},
                 "published": {"sparsegpt", "wanda", "flap"}}
    family, method = cfg.benchmark.family, cfg.branch.name
    comparison = cfg.protocol.get("refit_comparison")
    if comparison and (family != "opt" or method != "discrete_simple" or not comparison.source or
                       not 0 < comparison.target < 1 or comparison.proposal_index < 0 or
                       not 0 < comparison.rtol < 1 or comparison.max_iterations < 1):
        raise ValueError("refit_comparison requires OPT discrete_simple, a source run, and valid solver settings")
    if family not in supported or method not in supported[family]:
        raise ValueError(f"unsupported benchmark/method: {cfg.benchmark.name}/{method}")
    if family == "local" and method == "sparsegpt" and cfg.benchmark.name != "modadd":
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
    fixed_l0 = method == "discrete_simple" and cfg.branch.options.get("penalty_mode") == "fixed"
    if ((method in {"classic", "discrete"} and schedule.mode != "greedy") or
            (schedule.mode == "greedy" and method not in {"classic", "discrete"} and not fixed_l0)):
        raise ValueError("greedy schedules require classic, discrete, or fixed-penalty discrete_simple")
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
    if family == "network":
        from pruning.models.network import NetworkProblem
        problem = NetworkProblem(cfg)
    elif family == "opt":
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
    elif name == "discrete_simple":
        from pruning.methods.discrete_simple import DiscreteSimple
        method = DiscreteSimple(problem, options)
    elif name == "discrete":
        from pruning.methods.discrete import Discrete
        method = Discrete(problem, options)
    elif family == "local":
        from pruning.methods.local import LocalGreedy
        method = LocalGreedy(cfg, options)
    elif name == "continuous":
        from pruning.methods.continuous import Continuous
        method = Continuous(problem, options)
    elif name == "canonical_sparsegpt":
        from pruning.methods.canonical_sparsegpt import CanonicalSparseGPT
        method = CanonicalSparseGPT(problem, options)
    elif name == "classic":
        from pruning.methods.classic import OptClassic, MnistClassic
        method = (OptClassic if family == "opt" else MnistClassic)(options, cfg.protocol)
    else:
        from pruning.baselines.language import LanguageBaseline
        method = LanguageBaseline(name, options)
    metadata = {"benchmark": cfg.benchmark.name, "method": name, "device": cfg.runtime.device,
                "schedule": OmegaConf.to_container(cfg.branch.schedule, resolve=True),
                "protocol": OmegaConf.to_container(cfg.protocol, resolve=True),
                "method_options": OmegaConf.to_container(options, resolve=True)}
    if name == "discrete":
        metadata.update(metric="full Fisher; streamed exact classification expectations",
                        parameter_accounting="explicit coordinate masks after exact canonicalization",
                        canonical_chains=[list(chain.weights) for chain in method.chains])
    if name == "discrete_simple":
        metadata.update(metric="block-diagonal affine K-FAC", algorithm="L0 proximal gradient with support refit",
                        objective="0.5 * (V-W).T F (V-W) + penalty * canonical_remaining(V)",
                        compensation="unbounded affine coefficients; full support reconsidered at every IHT step",
                        budget_allocation="one common L0 penalty; sparsity targets are hints only",
                        canonicalization="exact two-affine cleanup" if method.chains else "none",
                        canonical_chains=[list(chain.weights) for chain in method.chains],
                        quadratic_evaluation="before function-preserving canonicalization",
                        progress_unit="linear_module",
                        pruning_parameters=list(method.initial_nonzero),
                        parameter_accounting="zeros introduced relative to the original nonzero coefficients")
        if options.get("penalty_mode") == "fixed":
            metadata.update(budget_allocation="one global penalty, calibrated once and then fixed; no module quotas",
                            permanent_deletions=options.get("permanent_deletions", False),
                            initial_sparsity_hint=options.get("initial_sparsity", .05))
    if family == "opt":
        s = cfg.sampling
        metadata.update(model=cfg.benchmark.model, model_dtype=cfg.benchmark.dtype,
                        sampling=OmegaConf.to_container(s, resolve=True),
                        parameter_accounting="decoder_linear_weights_and_biases",
                        total_model_parameters=sum(parameter.numel() for parameter in problem.model.parameters()),
                        pruning_modules=list(problem.names), accounting_modules=list(problem.accounting_names),
                        deletion_unit=options.get("deletion_unit"), calibration={
                            "dataset": s.dataset, "nsamples": s.calibration_sequences,
                            "seqlen": s.context, "seed": s.seed, "refresh": s.refresh,
                            "b_sequences": s.b_sequences if name in {"continuous", "discrete_simple"} else None,
                            "b_context": s.b_context if name in {"continuous", "discrete_simple"} else None,
                            "b_tokens": problem.settings.b_tokens if name in {"continuous", "discrete_simple"} else None})
        if name == "canonical_sparsegpt":
            metadata.update(metric="A-based SparseGPT reconstruction", canonicalization="exact MLP cleanup",
                            completion_refresh="once per progressive proposal",
                            completion_rule="score *= (remaining_nonzero / group_width) ** completion_power",
                            compensation="SparseGPT sequential propagation; existing zeros remain fixed",
                            fc2_statistics="refreshed after fc1 pruning and canonicalization")
    elif family == "local":
        from pruning.models.local import checkpoint_sha256
        metadata["checkpoint_sha256"] = checkpoint_sha256(cfg.benchmark.name)
    recorder = Recorder(run, metadata)
    if cfg.protocol.get("refit_comparison"):
        from pruning.diagnostics import compare_l0_refit
        return compare_l0_refit(problem, method, cfg, recorder)
    return run_path(problem, method, cfg.protocol, cfg.branch.schedule, recorder)
