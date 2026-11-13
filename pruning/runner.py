from __future__ import annotations

import random
import torch
from omegaconf import OmegaConf
from pruning.engine import run_path
from pruning.results import Recorder


def validate(cfg):
    supported = {"local": {"classic", "berman", "sparsegpt", "discrete", "reference"},
                 "mnist": {"classic", "discrete", "discrete_simple", "continuous", "berman", "reference"},
                 "network": {"discrete", "discrete_simple", "continuous", "reference"},
                 "opt": {"classic", "continuous", "canonical_sparsegpt", "discrete_simple", "sparsegpt", "wanda", "flap",
                         "reference", "native"},
                 "llama": {"continuous", "discrete_simple", "discrete", "reference", "native"}}
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
    greedy_baseline = method == "reference" and cfg.branch.options.score == "mean_ablation"
    if ((method in {"classic", "discrete"} and schedule.mode != "greedy") or
            (schedule.mode == "greedy" and method not in {"classic", "discrete"} and not fixed_l0 and not greedy_baseline)):
        raise ValueError("greedy schedules require a greedy method or baseline")
    if greedy_baseline and schedule.mode != "greedy":
        raise ValueError("Structural baselines require a greedy schedule")
    if method == "berman" and cfg.branch.options.get("scheme", "iterative") in {"one_shot", "magnitude", "fim"} and schedule.mode != "independent":
        raise ValueError("Berman's static schemes require an independent schedule")
    if method == "reference" and cfg.branch.options.get("refresh") == "step" and schedule.mode != "progressive":
        raise ValueError("Refreshed ranked baselines require a progressive schedule")
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
    if name == "native":
        from pruning.baselines.native import NativeMethod, NativeProblem
        problem = NativeProblem(cfg, run)
        recorder = Recorder(run, {"benchmark": cfg.benchmark.name, "method": cfg.branch.options.program,
                                  "label": cfg.branch.label, "device": cfg.runtime.device, **problem.metadata,
                                  "schedule": OmegaConf.to_container(cfg.branch.schedule, resolve=True),
                                  "protocol": OmegaConf.to_container(cfg.protocol, resolve=True)})
        return run_path(problem, NativeMethod(problem), cfg.protocol, cfg.branch.schedule, recorder)
    options = OmegaConf.merge(
        OmegaConf.to_container(cfg.benchmark.get("methods", {}).get(name, OmegaConf.create({})), resolve=True),
        OmegaConf.to_container(cfg.branch.options, resolve=True))
    if family == "network":
        from pruning.models.network import NetworkProblem
        problem = NetworkProblem(cfg)
    elif family in {"opt", "llama"}:
        from pruning.models.opt import OptProblem
        problem = OptProblem(cfg, options)
    elif family == "mnist":
        from pruning.models.mnist import MnistProblem
        problem = MnistProblem(cfg)
    else:
        from pruning.models.local import LocalProblem
        problem = LocalProblem(cfg)
    if name == "reference":
        from pruning.baselines.reference import ReferenceBaseline
        method = ReferenceBaseline(problem, options)
    elif name == "berman":
        from pruning.baselines.berman import Berman
        method = Berman(options)
    elif family == "local" and name == "sparsegpt":
        from pruning.baselines.modadd import ModaddSparseGPT
        method = ModaddSparseGPT(options)
    elif name == "discrete_simple":
        if options.get("factor_backend") == "streaming":
            from pruning.methods.discrete_simple_streaming import StreamingDiscreteSimple
            method = StreamingDiscreteSimple(problem, options)
        else:
            from pruning.methods.discrete_simple import DiscreteSimple
            method = DiscreteSimple(problem, options)
    elif name == "discrete":
        from pruning.methods.discrete import Discrete
        method = Discrete(problem, options)
    elif family == "local":
        from pruning.methods.local import LocalGreedy
        method = LocalGreedy(cfg, options)
    elif name == "continuous":
        if family in {"opt", "llama"}:
            if options.get("factor_backend") == "streaming":
                from pruning.methods.continuous_streaming import StreamingContinuous
                method = StreamingContinuous(problem, options)
            else:
                from pruning.methods.continuous import Continuous
                method = Continuous(problem, options)
        else:
            from pruning.methods.continuous_affine import AffineContinuous
            method = AffineContinuous(problem, options)
    elif name == "canonical_sparsegpt":
        from pruning.methods.canonical_sparsegpt import CanonicalSparseGPT
        method = CanonicalSparseGPT(problem, options)
    elif name == "classic":
        from pruning.methods.classic import OptClassic, MnistClassic
        method = (OptClassic if family == "opt" else MnistClassic)(options, cfg.protocol)
    else:
        from pruning.baselines.language import LanguageBaseline
        method = LanguageBaseline(name, options)
    metadata = {"benchmark": cfg.benchmark.name, "method": name, "label": cfg.branch.label, "device": cfg.runtime.device,
                "schedule": OmegaConf.to_container(cfg.branch.schedule, resolve=True),
                "protocol": OmegaConf.to_container(cfg.protocol, resolve=True),
                "method_options": OmegaConf.to_container(options, resolve=True)}
    if name == "discrete":
        metadata.update(metric="full Fisher; streamed exact classification expectations",
                        parameter_accounting="explicit coordinate masks after exact canonicalization",
                        canonical_chains=[list(chain.weights) for chain in method.chains])
    if name == "discrete_simple":
        fast = method.solver == "fast"
        metadata.update(metric="block-diagonal affine K-FAC", solver_backend=method.solver,
                        algorithm=("canonical L0 IHT with capped diagonal-CG refit" if fast else
                                   "canonical L0 proximal backtracking with checked AXB refit"),
                        objective="0.5 * (V-W).T F (V-W) + penalty * canonical_remaining(V)",
                        compensation="unbounded affine coefficients; full support reconsidered at every IHT step",
                        budget_allocation="one common L0 penalty; sparsity targets are hints only",
                        canonicalization="exact inferred-chain/gated cleanup" if method.chains or method.gated_groups else "none",
                        canonical_chains=[list(chain.weights) for chain in method.chains],
                        canonical_gated_groups=[list(group.names) for group in method.gated_groups],
                        quadratic_evaluation="before function-preserving canonicalization",
                        progress_unit="linear_module",
                        pruning_parameters=list(method.initial_nonzero),
                        parameter_accounting="zeros introduced relative to the original nonzero coefficients")
        if options.get("penalty_mode") == "fixed":
            metadata.update(budget_allocation="one global penalty, calibrated once and then fixed; no module quotas",
                            permanent_deletions=options.get("permanent_deletions", False),
                            initial_sparsity_hint=options.get("initial_sparsity", .05))
        metadata.update(proposal_initialization="accepted affine coefficients; no ranked mask",
                        support_refit=("capped diagonal-preconditioned CG; residual recorded; convergence not required" if fast else
                                       "inverse-AXB preconditioned CG; true residual checked; failure raises"),
                        proximal_steps=("power-iteration curvature; majorization backtracking; one final support refit" if fast else
                                         "adaptive quadratic-majorization backtracking; retry after support refit"))
        metadata["canonical_proximal_map"] = ("exact disjoint four-way gated groups" if method.gated_groups else
                                               "layer-coordinate approximation on overlapping chains"
                                               if any(len(chain.weights) > 2 for chain in method.chains)
                                               else "exact disjoint two-affine group comparisons")
    if name == "continuous" and options.get("affine", False):
        metadata.update(metric="block-diagonal affine K-FAC; float64 A/B moments",
                        algorithm="global singleton OBS selection with full affine complement compensation",
                        compensation="all live affine weights and biases; explicit deletions remain fixed",
                        pruning_parameters=list(method.initial_nonzero),
                        deletion_accounting="explicit direct and canonical masks; initially zero coefficients receive no removal credit",
                        canonicalization="exact MLP cleanup after each proposal" if options.get("canonicalization") else "none",
                        canonical_incentive="none; singleton OBS scores are unchanged",
                        ranking_approximation="unrestricted local inverse scores; joint batch and frozen-support interactions remain approximate",
                        protected_biases="fc2 folding targets" if options.get("canonicalization") else "none")
        if family != "opt":
            metadata.update(protected_biases="downstream folding targets",
                            factor_estimation="same sampled model-Fisher K-FAC estimator as discrete_simple")
    if name == "reference":
        metadata.update(reference="NNpruning working notes and archived core.py",
                        ranking=options.score, score_refresh=options.refresh,
                        compensation="current mean folding only" if options.score == "mean_ablation" else "none",
                        implementation="global Wanda-style ranking, not published rowwise Wanda" if options.score == "wanda" else options.score)
        if hasattr(method, "pruning_parameters"):
            metadata["pruning_parameters"] = method.pruning_parameters
    if family in {"opt", "llama"}:
        s = cfg.sampling
        metadata.update(model=cfg.benchmark.model, model_dtype=cfg.benchmark.dtype,
                        sampling=OmegaConf.to_container(s, resolve=True),
                        parameter_accounting="decoder_linear_weights_and_biases",
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
