# Pruning experiments

Forge is the experiment entry point. Each benchmark and method is a Hydra setting in `conf/`; `train.py` dispatches it and Forge stores the resolved configuration, run metadata, per-step logs, final metrics, and result files under `outputs/xps/`.

## Setup

Use Python 3.10 or later. The dependency versions in `requirements.txt` match the machine used for the current OPT and Berman work. Install Forge as a sibling checkout:

```bash
git clone https://github.com/simonelusetti/pruning.git
git clone https://github.com/simonelusetti/forge.git
cd pruning
git switch fisherT
python3 -m pip install -r requirements.txt
python3 -m pip install -e ../forge
```

The setup was validated against local Forge commit `6ede2b5`, which includes fixes to runtime-key signatures and grid failure reporting. That commit is awaiting publication in the separate Forge repository; until then, a fresh Forge clone has different behavior for those two features.

For `fisher_general` and `fisher_miqp`, also install `gurobipy` and provide a working Gurobi license. The local moons and modular-addition benchmarks generate their own data. OPT requires access to the `facebook/opt-125m` checkpoint and C4 data through Hugging Face; model and dataset caches are not tracked. The Berman/MNIST comparison requires the released SimpleNN checkpoint at `checkpoints/berman/simplenn_mnist_official.pt` and the four standard MNIST gzip files in `native_baselines/cache/berman/mnist/`. Those inputs are not bundled with this repository.

The [supervisor report](reports/supervisor_progress.md) summarizes methods and evidence. The [implementation reference](reports/implementation_reference.md) gives exact candidate rules, objectives, settings, and comparison limits.

## Run experiments

```bash
forge run
forge run benchmark=moons branch=fisher_general runtime.device=cpu
forge run benchmark=modadd branch=classic runtime.device=cuda:0
forge run benchmark=modadd_berman branch=berman runtime.device=cpu
forge run benchmark=berman branch=fisher_sparsegpt_berman runtime.device=cpu
forge run benchmark=opt branch=fisher_sparsegpt runtime.device=cuda:0
forge run benchmark=opt branch=fisher_complement runtime.device=cuda:0
forge run benchmark=opt branch=fisher_complement_refresh runtime.device=cpu
forge run benchmark=opt_cpu branch=sparsegpt runtime.device=cpu
```

Use `forge grid` for comparisons, for example:

```bash
forge grid benchmark=berman runtime.device=cpu --run branch=berman --run branch=classic --run branch=fisher_general
forge metrics benchmark.name=berman
forge artifact 'result.json'
```

Benchmark and branch preset names ending in `_berman` or `_modadd` supply benchmark-specific controls. Their `benchmark.name` and `branch.name` still identify the shared task and method. Set experiment parameters under `benchmark.*` or `branch.options.*`; `runtime.device` and `runtime.threads` control execution and do not change Forge's experiment signature. OPT random seeds are in `benchmark.seed` because they affect the experiment. A run with an unsupported benchmark/method pair fails before model loading.

`fisher_sparsegpt` is the maintained OPT Fisher-aware SparseGPT branch. `fisher_complement` uses the same factorized Fisher geometry with generalized-OBS deletion and full same-layer quadratic compensation. `fisher_miqp` uses a bounded same-layer mixed-integer pool. `fisher_general` is the Gurobi MLP method. Classic structural and weight-zeroing moves use `classic` and `classic_weights`. Berman, SparseGPT, Wanda, and FLAP baselines are settings as well. The published branches launch the pinned authors' implementations in their separate environments and require CUDA; bootstrap those repositories with `python3 native_baselines/bootstrap.py` first.

Result files live inside each Forge run. Plot completed curves with:

```bash
python3 plot_results.py berman
python3 plot_results.py opt
```

These plots use only completed Forge runs. Historical JSON files predating Forge were not imported because their full configuration and artifact provenance could not be reconstructed exactly. Existing static figures remain in `plots/` as historical snapshots.

Algorithm code is in `NNpruning/nnprune/` and `pruning_experiments/`. The latter contains benchmark implementations called by the Forge dispatcher; they are not separate experiment entry points. `native_baselines/` retains vendor bootstrap, SparseGPT primitives, and pinned source specifications.
