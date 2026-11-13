"""Authors' programs as methods of the shared engine.

Each proposal runs the pinned program at one target from the dense model and saves its output. `native_eval.py`
then scores that output on our token windows inside the program's own environment, so validation stopping,
held-out loss, parameter accounting, and inference measurement follow the same protocol as every other method."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from pruning.baselines.specs import SPECS
from pruning.models.opt import language_splits
from pruning.paths import BASELINE_CACHE, BASELINE_VENDOR, MODEL_CACHE, ROOT

COMPAT = Path(__file__).resolve().parent / "compat"
EVALUATE = Path(__file__).resolve().parent / "native_eval.py"
NUMBER = r"([0-9.eE+-]+|nan|inf)"
# The programs' own perplexity reports, kept as a cross-check on our evaluation of their outputs.
REPORTED = {"wanda": "wikitext perplexity " + NUMBER,
            "flap": "ppl on wikitext " + NUMBER, "slicegpt": "After rotating and slicing " + NUMBER,
            "llm_pruner": r"PPL after pruning: \{'wikitext2': " + NUMBER}


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          check=True, text=True, capture_output=True).stdout.strip()


def package_versions(python: str):
    code = ("import importlib.metadata as m,json; out={}\n"
            "for x in ['torch','transformers','datasets','accelerate','sentencepiece','safetensors']:\n"
            " try: out[x]=m.version(x)\n except m.PackageNotFoundError: out[x]=None\n"
            "print(json.dumps(out))")
    try:
        return json.loads(subprocess.run([python, "-c", code], check=True, text=True, capture_output=True).stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def version_mismatches(program: str, versions):
    if versions is None:
        return ["could not inspect packages"]
    return [f"{name}={versions.get(name)!r} (expected {expected})"
            for name, expected in SPECS[program]["versions"].items()
            if versions.get(name) is None or not versions[name].startswith(expected)]


def build_command(program, python, repo, model, target, seed, output) -> list[str]:
    spec = SPECS[program]
    script, nsamples = str(repo / spec["script"]), str(spec["nsamples"])
    if program == "wanda":
        return [python, script, "--model", model, "--seed", str(seed), "--nsamples", nsamples,
                "--sparsity_ratio", str(target), "--sparsity_type", "unstructured", "--prune_method", "wanda",
                "--cache_dir", str(MODEL_CACHE), "--save_model", str(output)]
    if program == "slicegpt":
        return [python, script, "--model", model, "--cal-dataset", spec["dataset"], "--cal-nsamples", nsamples,
                "--cal-max-seqlen", str(spec["seqlen"]), "--seed", str(seed), "--sparsity", str(target),
                "--device", "cuda:0", "--no-wandb", "--save-dir", str(output)]
    if program == "llm_pruner":
        layers = ["--block_mlp_layer_start", "4", "--block_mlp_layer_end", "30",
                  "--block_attention_layer_start", "4", "--block_attention_layer_end", "30"]
        return [python, script, "--base_model", model, "--pruning_ratio", str(target), "--device", "cpu",
                "--eval_device", "cuda", "--block_wise", *layers, "--save_ckpt_log_name", str(output),
                "--pruner_type", "taylor", "--taylor", "param_first", "--seed", str(seed), "--save_model"]
    return [python, script, "--model", model, "--seed", str(seed), "--nsamples", nsamples,
            "--pruning_ratio", str(target), "--remove_heads", "-1", "--metrics", "WIFV", "--structure", "AL-AM",
            "--prune_method", "flap", "--cache_dir", str(MODEL_CACHE), "--save_model", str(output)]


def run_and_tee(command, cwd, env, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


class NativeProblem:
    """The state is the latest scored output; restoring a state needs no model in this process."""

    def __init__(self, cfg, run):
        self.cfg, self.program = cfg, cfg.branch.options.program
        spec = SPECS[self.program]
        sampling = cfg.sampling
        if (cfg.benchmark.model != spec["model"] or sampling.dataset != spec["dataset"] or
                sampling.calibration_sequences != spec["nsamples"] or sampling.context != spec["seqlen"]):
            raise ValueError(f"{cfg.benchmark.name} does not reproduce {self.program}'s published model and calibration")
        if not str(cfg.runtime.device).startswith("cuda"):
            raise ValueError("native programs require runtime.device=cuda:N")
        self.repo = BASELINE_VENDOR / self.program
        head = git_head(self.repo)
        if head != spec["commit"]:
            raise RuntimeError(f"{self.program} source is {head}, expected {spec['commit']}; run bootstrap.py")
        self.python = str(ROOT / (cfg.runtime.python or spec["python"]))
        versions = package_versions(self.python)
        mismatches = version_mismatches(self.program, versions)
        if mismatches and not cfg.branch.options.allow_version_mismatch:
            raise RuntimeError("native environment mismatch: " + "; ".join(mismatches))
        self.env = os.environ.copy()
        paths = [str(self.repo / "src")] if self.program == "slicegpt" else [str(COMPAT), str(self.repo)]
        self.env.update(PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(cfg.runtime.device).partition(":")[2] or "0",
                        HF_HOME=str(BASELINE_CACHE), HF_DATASETS_CACHE=str(BASELINE_CACHE / "datasets"),
                        TRANSFORMERS_CACHE=str(MODEL_CACHE), HUGGINGFACE_HUB_CACHE=str(MODEL_CACHE),
                        PYTHONPATH=os.pathsep.join([*paths, *filter(None, [os.environ.get("PYTHONPATH")])]))
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], cache_dir=MODEL_CACHE,
                                                  use_fast=cfg.benchmark.get("tokenizer_fast", False))
        positions = AutoConfig.from_pretrained(spec["model"], cache_dir=MODEL_CACHE).max_position_embeddings
        self.splits, self.split_info = language_splits(cfg, tokenizer, positions)
        self.work = run.path / "native"
        self.work.mkdir(exist_ok=True)
        torch.save({name: torch.cat(values) for name, values in self.splits.items() if values}, self.work / "tokens.pt")
        self.metadata = {"source_commit": head, "package_versions": versions, "version_mismatches": mismatches,
                         "paper": spec["paper"], "model": spec["model"], "paper_model": spec.get("paper_model", spec["model"]),
                         "model_dtype": cfg.benchmark.dtype,
                         "parameter_accounting": "nonzero decoder-layer coefficients, norms excluded",
                         "calibration": {"dataset": spec["dataset"], "nsamples": spec["nsamples"], "seqlen": spec["seqlen"]}}
        self.dense = self.state = self.evaluate("dense", None, None, self.work / "dense.json")
        self.total = self.dense["pool_nonzero"]
        self.extra = {}

    def evaluate(self, kind, path, target, output):
        command = [self.python, str(EVALUATE), "--kind", kind, "--model", SPECS[self.program]["model"],
                   "--tokens", str(self.work / "tokens.pt"), "--cache", str(MODEL_CACHE),
                   "--dtype", self.cfg.benchmark.dtype, "--output", str(output),
                   "--batch-size", str(self.cfg.sampling.evaluation_batch_size)]
        command += ["--path", str(path), "--target", str(target)] if path is not None else []
        subprocess.run(command, cwd=self.repo, env=self.env, check=True)
        state = json.loads(output.read_text())
        state["losses"] = {name: torch.tensor(values, dtype=torch.float64) for name, values in state["losses"].items()}
        return state

    def snapshot(self):
        return self.state

    def restore(self, state):
        self.state = state

    def remaining(self):
        return self.state["pool_nonzero"]

    def outside_removed(self):
        return ((self.dense["model_nonzero"] - self.dense["pool_nonzero"]) -
                (self.state["model_nonzero"] - self.state["pool_nonzero"]))

    def model_parameters(self):
        return self.dense["model_parameters"]

    def parameter_bytes(self):
        return self.state["parameter_bytes"]

    def compact(self):
        pass

    def profile(self):
        return self.losses("reporting"), {"seconds": self.state["seconds"], "parameter_bytes": self.parameter_bytes()}

    def losses(self, split):
        return self.state["losses"][split]

    def split_metadata(self, split):
        return dict(self.split_info[split])

    def prepare(self, index):
        return {}


class NativeMethod:
    def __init__(self, problem):
        self.problem = problem

    def propose(self, problem, target, index, progress):
        step = problem.work / f"step_{index + 1}"
        output = step / "model"
        command = build_command(problem.program, problem.python, problem.repo, SPECS[problem.program]["model"],
                                target, problem.cfg.benchmark.seed, output)
        if run_and_tee(command, problem.repo, problem.env, step / "stdout.log"):
            raise RuntimeError(f"{problem.program} exited with an error at target {target}; see {step / 'stdout.log'}")
        problem.state = problem.evaluate(problem.program, output, target, step / "scores.json")
        shutil.rmtree(output)
        reported = re.findall(REPORTED[problem.program], (step / "stdout.log").read_text(encoding="utf-8"))
        return {"native_reported_ppl": float(reported[-1])} if reported else {}
