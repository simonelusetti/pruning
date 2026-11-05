from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Plan:
    module: str
    arguments: tuple[str, ...]
    artifact: str | None


def _flags(options: dict, mapping: dict[str, str]) -> list[str]:
    unknown = set(options) - set(mapping)
    if unknown:
        raise ValueError(f"unsupported branch settings: {sorted(unknown)}")
    result = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        flag = mapping[key]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        result.extend((flag,) if value is True else (flag, str(value)))
    return result


def build_plan(cfg: DictConfig, run_dir: Path) -> Plan:
    benchmark, branch = cfg.benchmark.name, cfg.branch.name
    options = OmegaConf.to_container(cfg.branch.options, resolve=True)
    common = ["--threads", str(cfg.runtime.threads)]
    device = ["--device", str(cfg.runtime.device)]
    output = ["--output", str(run_dir / "result.json")]
    artifact = "result.json"

    if benchmark == "moons" and branch == "berman":
        args = [*common, *device, *output,
                "--fisher-examples", str(cfg.benchmark.fisher_n),
                "--max-loss-error", str(cfg.benchmark.max_loss_error),
                "--validation-split-seed", str(cfg.benchmark.validation_split_seed)]
        return Plan("moons_berman", tuple(args), artifact)

    if benchmark == "modadd" and branch == "berman":
        if cfg.benchmark.fisher_n not in (500, 1740):
            raise ValueError("modular-addition Berman supports 500 or 1740 Fisher examples")
        args = [*common, *device, "--method", "berman", "--rho",
                str(cfg.benchmark.rho), "--n-sigma", str(cfg.benchmark.n_sigma), "--fisher-n",
                str(cfg.benchmark.fisher_n), "--output-dir", str(run_dir)]
        return Plan("modadd", tuple(args), f"berman_iterative_n{cfg.benchmark.fisher_n}.json")
    if benchmark == "modadd" and branch == "sparsegpt":
        args = [*common, *device, *output, "--calibration",
                str(cfg.benchmark.calibration)]
        return Plan("modadd_sparsegpt", tuple(args), artifact)

    if benchmark == "berman" and branch in {
            "berman", "classic", "classic_weights", "fisher_general", "fisher_sparsegpt"}:
        filename = "berman_iterative.json" if branch == "berman" else f"{branch}.json"
        args = [*common, *device, "--method", branch, "--rho", str(cfg.benchmark.rho),
                "--n-sigma", str(cfg.benchmark.n_sigma), "--fisher-n",
                str(cfg.benchmark.fisher_n), "--output-dir", str(run_dir)]
        if branch == "fisher_general":
            args += _flags(options, {key: f"--fisher-general-{key.replace('_', '-')}"
                                     for key in options})
        elif branch == "fisher_sparsegpt":
            args += _flags(options, {"targets": "--sparsegpt-targets",
                                     "blocksize": "--sparsegpt-blocksize",
                                     "percdamp": "--sparsegpt-percdamp",
                                     "b_damp": "--sparsegpt-b-damp"})
        elif options:
            raise ValueError(f"unsupported {branch} options: {sorted(options)}")
        return Plan("berman", tuple(args), filename)

    if benchmark == "berman" and branch == "matched_updates":
        args = [*common, *device, *output, "--fisher-n", str(cfg.benchmark.fisher_n),
                "--rho", str(cfg.benchmark.rho), "--n-sigma", str(cfg.benchmark.n_sigma),
                *_flags(options, {"targets": "--targets"})]
        return Plan("berman_matched", tuple(args), artifact)

    if benchmark in {"opt", "opt_cpu"} and branch in {"classic", "classic_weights"}:
        if benchmark == "opt" and (cfg.benchmark.rho != .05 or cfg.benchmark.n_sigma != 2.):
            raise ValueError("OPT Classic uses rho=0.05 and n_sigma=2 in this implementation")
        args = [*common, *device, *output, "--model", str(cfg.benchmark.model),
                "--nsamples", str(cfg.benchmark.nsamples), "--seqlen",
                str(cfg.benchmark.seqlen), "--eval-tokens", str(cfg.benchmark.eval_tokens),
                "--move-group", branch, "--seed", str(cfg.benchmark.seed),
                "--checkpoint", str(run_dir / "checkpoint.json"),
                "--model-checkpoint", str(run_dir / "model.pt")]
        return Plan("opt_classic", tuple(args), artifact)

    if benchmark == "opt_cpu" and branch in {"sparsegpt", "wanda", "flap"}:
        if str(cfg.runtime.device) != "cpu":
            raise ValueError("opt_cpu baselines require runtime.device=cpu")
        args = [branch, *common, *output, "--model", str(cfg.benchmark.model),
                "--nsamples", str(cfg.benchmark.nsamples), "--seed",
                str(cfg.benchmark.seed), "--seqlen",
                str(cfg.benchmark.seqlen), "--eval-tokens", str(cfg.benchmark.eval_tokens),
                *_flags(options, {"sparsity": "--sparsity"})]
        return Plan("opt_cpu", tuple(args), artifact)

    if benchmark == "opt" and branch in {"fisher_sparsegpt", "fisher_complement", "fisher_miqp"}:
        preset = "fisher_sparsegpt"
        args = [*common, *device, *output, "--model",
                str(cfg.benchmark.model), "--nsamples", str(cfg.benchmark.nsamples),
                "--seqlen", str(cfg.benchmark.seqlen), "--rho", str(cfg.benchmark.rho),
                "--n-sigma", str(cfg.benchmark.n_sigma), "--seed", str(cfg.benchmark.seed),
                "--fisher-preset", preset]
        if branch == "fisher_miqp":
            args += ["--fisher-deletion", "miqp", "--fisher-compensation", "sparse_miqp",
                     "--fisher-update-scope", "layer"]
        mapping = {"targets": "--targets", "geometry": "--fisher-geometry",
                   "model_dtype": "--model-dtype",
                   "deletion": "--fisher-deletion", "compensation": "--fisher-compensation",
                   "structure": "--fisher-structure", "decision": "--fisher-decision",
                   "mu": "--fisher-mu", "pool": "--fisher-pool",
                   "repeat": "--fisher-repeat", "time_limit": "--fisher-time-limit",
                   "gap": "--fisher-gap", "b_sequences": "--b-sequences",
                   "b_context": "--b-context", "b_tokens": "--b-tokens",
                   "b_batch_size": "--b-batch-size", "b_damp_ratio": "--b-damp-ratio",
                   "b_refresh": "--b-refresh",
                   "cg_rtol": "--fisher-cg-rtol",
                   "cg_max_iterations": "--fisher-cg-max-iterations"}
        args += _flags(options, mapping)
        return Plan("opt_fisher", tuple(args), artifact)

    if benchmark == "opt" and branch == "matched_updates":
        if (cfg.benchmark.seed != 0 or cfg.benchmark.model != "facebook/opt-125m" or
                cfg.benchmark.nsamples != 128 or cfg.benchmark.seqlen != 2048 or
                cfg.benchmark.rho != .05 or cfg.benchmark.n_sigma != 2.):
            raise ValueError("matched OPT updates use fixed OPT-125M, 128x2048 C4, seed 0")
        args = [*common, *device, *output,
                *_flags(options, {"target_sparsity": "--target-sparsity",
                                  "heldout_count": "--heldout-count", "mus": "--mus",
                                  "sequential_update": "--sequential-update"})]
        return Plan("opt_matched", tuple(args), artifact)

    if benchmark == "opt" and branch == "b_sweep":
        if (cfg.benchmark.seed != 0 or cfg.benchmark.nsamples != 128 or
                cfg.benchmark.seqlen != 2048 or cfg.benchmark.rho != .05 or
                cfg.benchmark.n_sigma != 2.):
            raise ValueError("B sweep uses fixed 128x2048 C4 calibration and seed 0")
        args = [*common, *device, *output, "--model", str(cfg.benchmark.model),
                *_flags(options, {"cases": "--cases", "heldout_count": "--heldout-count"})]
        return Plan("opt_b_sweep", tuple(args), artifact)

    if benchmark == "opt" and branch == "sparsegpt_curve":
        if (str(cfg.runtime.device) != "cpu" or cfg.benchmark.seed != 0 or
                cfg.benchmark.model != "facebook/opt-125m" or
                cfg.benchmark.nsamples != 128 or cfg.benchmark.seqlen != 2048):
            raise ValueError("SparseGPT published curve uses CPU, OPT-125M, 128x2048 C4, seed 0")
        args = [*common, *output, "--rho", str(cfg.benchmark.rho),
                "--n-sigma", str(cfg.benchmark.n_sigma),
                *_flags(options, {"fine_spacing": "--fine-spacing",
                                  "fine_until": "--fine-until",
                                  "coarse_spacing": "--coarse-spacing",
                                  "max_sparsity": "--max-sparsity"})]
        return Plan("opt_sparsegpt_curve", tuple(args), artifact)

    if benchmark in {"berman", "opt"} and branch == "compensation_ablation":
        if benchmark == "berman" and (cfg.benchmark.fisher_n != 500 or
                                      cfg.benchmark.rho != .05 or cfg.benchmark.n_sigma != 2.):
            raise ValueError("Berman compensation ablation uses 500 Fisher images")
        if benchmark == "opt" and (
                cfg.benchmark.model != "facebook/opt-125m" or cfg.benchmark.seed != 0 or
                cfg.benchmark.nsamples != 128 or cfg.benchmark.seqlen != 2048 or
                cfg.benchmark.rho != .05 or cfg.benchmark.n_sigma != 2.):
            raise ValueError("OPT compensation ablation uses fixed OPT-125M, 128x2048 C4, seed 0")
        args = [*common, *device, *output, "--setting", benchmark,
                *_flags(options, {"target": "--target", "mu": "--mu"})]
        return Plan("compensation", tuple(args), artifact)

    if benchmark == "published" and branch in {
            "sparsegpt_published", "wanda_published", "flap_published"}:
        method = branch.removesuffix("_published")
        from native_baselines.specs import SPECS
        spec = SPECS[method]
        if (cfg.benchmark.source != method or cfg.benchmark.model != spec["model"] or
                cfg.benchmark.dataset != spec["dataset"] or
                cfg.benchmark.nsamples != spec["nsamples"]):
            raise ValueError("published benchmark must match its pinned model and data settings")
        if not str(cfg.runtime.device).startswith("cuda"):
            raise ValueError("published native implementations require runtime.device=cuda:N")
        gpu = str(cfg.runtime.device).partition(":")[2] or "0"
        args = [method, "--execute", "--output-root", str(run_dir),
                "--seed", str(cfg.benchmark.seed), "--gpu", gpu,
                "--model", str(cfg.benchmark.model), "--dataset", str(cfg.benchmark.dataset),
                "--nsamples", str(cfg.benchmark.nsamples),
                *_flags(options, {"sparsity": "--sparsity", "python": "--python"})]
        return Plan("native", tuple(args), None)

    raise ValueError(f"unsupported benchmark/branch: {benchmark}/{branch}")
