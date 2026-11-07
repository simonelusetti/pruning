# Pruning experiments

Forge is the experiment entry point. Each benchmark and method is a Hydra setting in `conf/`; `train.py` dispatches it and Forge stores the resolved configuration, run metadata, per-step logs, final metrics, and result files under `outputs/xps/`. Maintained code lives in `pruning/`. All local checkpoints, caches, figures, reports, and run artifacts live under `outputs/`.

## Setup

Use Python 3.10 or later. The dependency versions in `requirements.txt` match the machine used for the current OPT and Berman work. Install Forge as a sibling checkout:

```bash
git clone https://github.com/simonelusetti/pruning.git
git clone https://github.com/simonelusetti/forge.git
cd pruning
python3 -m pip install -r requirements.txt
python3 -m pip install -e ../forge
```

For `fisher_general`, also install `gurobipy` and provide a working Gurobi license. The local moons and modular-addition benchmarks generate their own data. OPT requires access to the `facebook/opt-125m` checkpoint and C4 data through Hugging Face; model and dataset caches are not tracked. The Berman/MNIST comparison requires the released SimpleNN checkpoint at `outputs/checkpoints/berman/simplenn_mnist_official.pt` and the four standard MNIST gzip files in `outputs/cache/native_baselines/berman/mnist/`. Those inputs are not bundled with this repository.

The [supervisor report](outputs/reports/supervisor_progress.md) summarizes methods and evidence. The [implementation reference](outputs/reports/implementation_reference.md) gives exact candidate rules, objectives, settings, and comparison limits.

## Run experiments

```bash
forge run
forge run benchmark=moons branch=fisher_general runtime.device=cpu
forge run benchmark=modadd branch=classic runtime.device=cuda:0
forge run benchmark=modadd_berman branch=berman runtime.device=cpu
forge run benchmark=opt branch=fisher_sparsegpt runtime.device=cuda:0
forge run benchmark=opt branch=fisher_sparsegpt_refresh runtime.device=cpu
forge run benchmark=opt branch=fisher_complement_progressive runtime.device=cpu
forge run benchmark=opt_cpu branch=sparsegpt runtime.device=cpu
```

Use `forge grid` for comparisons, for example:

```bash
forge grid benchmark=berman runtime.device=cpu --run branch=berman --run branch=classic --run branch=fisher_general
forge metrics benchmark.name=berman
forge artifact 'result.json'
```

Benchmark and branch preset names ending in `_berman` or `_modadd` supply benchmark-specific controls. Their `benchmark.name` and `branch.name` still identify the shared task and method. Set experiment parameters under `benchmark.*` or `branch.options.*`; `runtime.device` and `runtime.threads` control execution and do not change Forge's experiment signature. OPT random seeds are in `benchmark.seed` because they affect the experiment. A run with an unsupported benchmark/method pair fails before model loading.

`fisher_sparsegpt` is the maintained OPT Fisher-aware SparseGPT branch. The progressive complement preset uses its blockwise deletion mask, discards the temporary sequential updates, and solves the full same-module AXB quadratic for all surviving weights after each accepted sparsity step. `fisher_general` is the Gurobi MLP method. Classic structural and weight-zeroing moves use `classic` and `classic_weights`. Berman, SparseGPT, Wanda, and FLAP baselines are settings as well. The published branches launch the pinned authors' implementations in their separate environments and require CUDA; bootstrap those repositories with `python3 -m pruning.baselines.bootstrap` first.

Result files live inside each Forge run. Plot the current Forge comparisons with:

```bash
python3 -m pruning.plot_results berman
python3 -m pruning.plot_results opt
```

The OPT plot reads completed Forge runs from the current comparison. Historical JSON files predating Forge were not imported because their full configuration and artifact provenance could not be reconstructed exactly. Existing static figures remain in `outputs/plots/` as historical snapshots.

`pruning/nnprune/` contains algorithms, `pruning/experiments/` contains benchmark implementations called by Forge, and `pruning/baselines/` contains pinned source specifications and local baseline primitives. Direct runner defaults write to `outputs/manual/`; Forge writes to `outputs/xps/`. Only curated figures and reports under `outputs/` are tracked by Git.
