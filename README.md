# Pruning experiments

`train.py` is the single experiment entry point, called by Forge. Hydra configs select the benchmark, method, sampling, and evaluation protocol. Internal methods run through direct Python calls and one shared execution loop. Forge owns configuration identity, run status, scalar logs, metrics, and artifacts under `outputs/xps/`.

## Setup

Use Python 3.10 or later and install Forge as a sibling checkout:

```bash
git clone https://github.com/simonelusetti/pruning.git
git clone https://github.com/simonelusetti/forge.git
cd pruning
python3 -m pip install -r requirements.txt
python3 -m pip install -e ../forge
```

`discrete` additionally requires `gurobipy` and a working license. Two-moons and modular addition generate their data and reuse checkpoints in `outputs/checkpoints/`. OPT downloads `facebook/opt-125m` and C4 through Hugging Face. The historical Berman/MNIST comparison needs `outputs/checkpoints/berman/simplenn_mnist_official.pt` and the four MNIST gzip files in `outputs/cache/native_baselines/berman/mnist/`; these inputs are not bundled.

## Running

```bash
forge run
forge run benchmark=moons branch=discrete runtime.device=cpu
forge run benchmark=modadd branch=classic runtime.device=cuda:0
forge run benchmark=modadd_berman branch=berman
forge run benchmark=opt branch=continuous
forge run benchmark=opt branch=continuous_global sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=continuous_neuron
forge run benchmark=opt branch=sparsegpt_curve
forge run benchmark=opt branch=canonical_sparsegpt sampling.calibration_sequences=32
forge run benchmark=opt_cpu branch=wanda
```

Launch long runs inside detached tmux sessions and verify the worker remains active. Work on `main`; leave changes uncommitted unless explicitly instructed otherwise.

The three framework methods are:

- `classic`: structural boundary moves, optionally combined with scalar zeroing using `branch=classic_weights`.
- `discrete`: bounded candidate pools, jointly optimized deletions and sparse updates by Gurobi MIQP, with canonicalization savings rewarded. Maintained on Bernoulli MLPs, with the historical SimpleNN/MNIST implementation retained.
- `continuous`: progressive OPT pruning. A blockwise generalized-OBS heuristic selects a deletion mask; temporary columnwise updates used during selection are discarded. The surviving weights in each linear module then receive a full-complement AXB quadratic solve. “Continuous” describes this fixed-mask update, not an exact optimization of the deletion set.

`continuous_global` is a preset of `continuous`. Once per progressive proposal it estimates all selected modules' A and B factors on the unchanged accepted model. It ranks surviving weights across all those modules using `W[i,j]² / (2 * B_inverse[i,i] * A_inverse[j,j])`, then selects exactly the additional deletions needed for the cumulative target. There are no per-block, per-module, or column-chunk quotas, and no temporary SparseGPT propagation. The same local full-complement solver handles compensation, using the snapshot's factors. Biases remain fixed and canonicalization is not applied. Cross-module interactions and joint deletion-set optimization remain approximations.

`continuous_neuron` is a preset of `continuous`, with `branch.options.deletion_unit=hidden_neuron`. It considers hidden neurons between `fc1` and `fc2`. Each neuron offers an incoming-row-plus-bias cut or an outgoing-column cut. Grouped compensated scores choose a direction and rank neurons globally by predicted damage per canonical parameter saving. A batch receives the full surviving-MLP quadratic update; function-preserving canonicalization removes the redundant counterpart. Ranking singleton groups into a batch remains heuristic, and cross-module Fisher interactions are omitted.

Berman, SparseGPT, Wanda, and FLAP are baselines, selected through the same entry point. Local OPT baseline implementations use the common execution loop; pinned published programs retain their authors' model/data settings and separate environments. See [baseline setup](pruning/baselines/README.md).

`canonical_sparsegpt` is an experimental SparseGPT-derived method. Once per progressive proposal it multiplies MLP deletion scores by `(remaining_nonzero / group_width) ** completion_power`, using incoming-row occupancy for fc1 and outgoing-column occupancy for fc2. The default power is 1; power 0 disables the bonus for a matched control. Attention uses ordinary SparseGPT scores. This uses A-based scoring and SparseGPT updates, with no B estimation or full-complement solve. Deleted coordinates remain fixed. Exact MLP canonicalization runs after each MLP module; fc2 input statistics are recollected after fc1 pruning and cleanup. Explicit masks count direct and canonical removals once, including removed fc1 biases; changes to fc2 bias from constant folding are not counted as deletions. The power-0 control shares this cleanup, fixed-mask handling, and statistics refresh, so it is distinct from the untouched native SparseGPT baseline.

## Configuration

| Config | Responsibility |
| --- | --- |
| `benchmark.*` | Model/data definition, split sizes, seeds, model-specific method defaults |
| `branch.options.*` | Algorithm settings, overriding `benchmark.methods.<method>` |
| `branch.schedule.*` | Greedy, progressive, or independent runs; targets and spacing |
| `sampling.*` | Language-model calibration and evaluation sequence counts, contexts, seeds |
| `protocol.*` | Candidate selection, validation stopping, reporting cadence, final model saving |
| `runtime.*` | Device, CPU threads, external baseline interpreter; excluded from experiment identity |

`branch.schedule.mode=progressive` retains accepted changes. `independent` restores the dense model before each sparsity point. `greedy` proposes one candidate move at a time. All use the same evaluation, acceptance, rollback, and recording implementation.

For OPT, `branch.options.modules` restricts the linear modules being pruned. Optional `branch.options.accounting_modules` selects the percentage denominator and must include those modules; it defaults to the pruning scope. For example, `modules="fc1,fc2"` with `accounting_modules=all` prunes only MLPs but measures removal against all transformer-block linear parameters. Continuous weight targets are scaled to that denominator; results record both scopes.

For example, tune continuous sampling without editing code:

```bash
forge run benchmark=opt branch=continuous sampling.calibration_sequences=16 sampling.b_sequences=16 sampling.heldout_sequences=32
forge run benchmark=opt branch=continuous branch.schedule.targets='[0.05,0.10,0.15]' protocol.save_model=true
forge run benchmark=opt branch=continuous_neuron protocol.selection_check=false sampling.heldout_sequences=32
```

Language-model sampling controls:

| Purpose | Count | Context | Seed |
| --- | --- | --- | --- |
| A / activation estimation | `sampling.calibration_sequences` | `sampling.context` | `sampling.seed` |
| B estimation | `sampling.b_sequences` | `sampling.b_context` | Same calibration pool |
| Classic candidate comparison | `sampling.candidate_sequences` | `sampling.selection_context` | `sampling.candidate_seed` |
| Validation acceptance | `sampling.selection_sequences` | `sampling.selection_context` | `sampling.selection_seed` |
| Independent reporting | `sampling.heldout_sequences` | `sampling.heldout_context` | `sampling.heldout_seed` |

OPT defaults to full 2,048-token contexts. A and B use independently sized prefixes of one calibration pool. `sampling.b_tokens=0` uses every valid gradient position: 128 sequences give 262,016 positions per module. A positive value explicitly caps gradient positions. `sampling.evaluation_batch_size` controls language-loss batching. Local supervised benchmarks use their configured example splits instead of these sequence settings.

`sampling.b_batch_size` and `sampling.activation_batch_size` control Fisher-backward and layer-replay batches, respectively; both default to 1 to bound memory use. Increasing them changes batching, not the sequence budget or token cap. `runtime.threads` caps PyTorch CPU parallelism (default 8); more threads can help full contexts and hurt small operations, so utilization alone is not a speed measure. B estimation accumulates double-precision covariances on the model device and stops backward propagation before the earliest requested block.

`sampling.refresh=step` redraws calibration once per pruning proposal, with seed `sampling.seed + proposal_index`; `fixed` reuses the initial draw. Continuous defaults to `step`; classic and baseline presets default to `fixed`. Default weight continuous refreshes B before each transformer block. Global-ranking and neuron continuous each collect one unchanged-model factor snapshot per proposal. Selection and reporting windows remain fixed throughout a run.

C4 sampling uses the first English training shard, chooses documents with replacement, rejects short documents, and draws contiguous windows. The shard loads once per process. Separate seeds do not guarantee document-disjoint samples; current C4 reporting uses training-shard windows, not the official validation split. Classic candidate and acceptance sets use separate windows. Every result records its actual split metadata and calibration seeds. Historical results retain their original settings.

`protocol.selection_check=true` accepts a proposal when its paired validation loss increase is at most `protocol.rho` times the original loss plus `protocol.n_sigma` standard errors. Rejection restores the accepted model. `false` skips this check and its data loading on OPT; no true-loss stopping guarantee remains. Classic candidate comparison is a separate choice: `protocol.candidate_selection=loss` or `proxy`. Reporting never chooses or accepts proposals.

Reporting runs every `protocol.reporting_every` proposals, default 1; 0 reports only the dense model and final accepted endpoint. The historical `opt_classic` preset uses endpoint reporting to avoid a full test evaluation for every small move. Standard errors use sequence means. `protocol.save_model=true` saves the final accepted state, including structural masks.

For neuron continuous, `protocol.trace=true` measures validation loss after each block's deletion, compensation, and canonicalization without changing its decisions. It saves `diagnostics.json`, the latest proposal's starting state, a rejected state if any, and the largest observed block jump with its factors. These additional full-context evaluations slow the run; tracing is disabled by default.

## Code layout and artifacts

```text
train.py                 Hydra + Forge lifecycle
pruning/runner.py        Resolve configs; construct model adapter and method
pruning/engine.py        Shared proposal/evaluation/acceptance/rollback loop
pruning/models/          Model loading, state snapshots, split evaluation, accounting
pruning/methods/         Candidate generation and pruning/update algorithms
pruning/nnprune/         Tensor solvers, canonicalization, local model primitives
pruning/data.py           Shared language-data sampling
pruning/curvature.py      Shared activation and Fisher-factor estimation
pruning/evaluation.py     Losses and paired statistics
pruning/results.py        Forge logs, result.json, optional model.pt
pruning/baselines/        Baseline algorithms and pinned external integrations
conf/                    All experiment presets
outputs/                 Runs, caches, checkpoints, figures, historical reports
```

There are no internal experiment subprocesses or per-method argument parsers. Subprocesses are used only to launch the authors' external baseline programs and inspect their pinned environments.

New internal OPT runs count decoder-linear weights and biases in a shared denominator; zero biases already present in the dense model are not credited as pruning. Historical CPU baseline artifacts used a weights-only sparsity field and retain that accounting. Neuron continuous counts structurally removed coordinates rather than incidental numerical zeros.

Each new internal run writes a common versioned `result.json`: `curve` contains evaluated proposals, `runs` includes in-progress method details, `accepted_endpoint` identifies the retained state, and `rejected_candidate` preserves the first failed validation proposal. `model.pt`, when requested, contains `state`, `target_sparsity`, and `removed_percent`; a PyTorch adapter's state contains `parameters` and `structure`, while local adapters retain their `State` object. Full configs and metrics are managed by Forge.

Post-hoc exact cleanup can save `canonicalization.json` and `model_canonicalized.pt` beside a source checkpoint. Plotting applies the correction only to the saved point identified in that artifact; original results and checkpoints remain unchanged. A star marks the cleaned endpoint and an open circle its original position. Earlier points require their own checkpoints to receive corrections.

Forge's `on_finish` command automatically refreshes `outputs/plots/<benchmark>/comparison.png` after a successful run. The default `[python3, pruning/plot_results.py]` is resolved by `train.py` to the current interpreter and an absolute script path before Forge changes directories. Set `forge.on_finish=null` to disable it, or supply a different command argument list. The command is excluded from experiment identity; plotting failure does not change a run's outcome.

`pruning/plots/opt.py`, `berman.py`, `two_moons.py`, and `modadd.py` own the benchmark layouts, sharing artifact loading and drawing helpers. They read `result.json` explicitly, so diagnostic JSON files do not hide runs. OPT shows the same held-out-loss curves in two panels: removal relative to each branch’s eligible parameter pool, and removal relative to all unique model parameters (shared weights counted once). Weight-wise pools include weights in the selected modules and exclude fixed biases; the neuron pool includes both MLP weight matrices and the removable fc1 biases, excluding fc2 biases. The eligible count is listed per curve; only the horizontal normalization changes between panels. New runs record the full-model count; older runs recover it from the locally cached model architecture without loading weights. It selects one saved configuration per branch and pruning scope: greatest accepted removal, then lowest validation damage, then newest run. MLP-only and all-module pruning remain separate comparison series. It groups compatible models, parameter denominators, evaluation datasets and contexts, and stopping rules; sequence counts may differ and are labelled. Every curve uses its own original-model loss, with held-out uncertainty shown separately. This is a comparison of selected configurations rather than a claim of matched evaluation budgets. First rejected proposals are marked with ×. Failed runs and running runs are excluded by default. Concurrent hooks serialize plot updates and replace images atomically. Published wrappers without a saved loss curve leave the existing image unchanged.

```bash
forge grid benchmark=moons --run branch=classic --run branch=classic_weights --run branch=discrete
forge metrics benchmark.name=moons
forge artifact 'result.json'
python3 -m pruning.plot_results opt
python3 -m pruning.plot_results berman
python3 -m pruning.plot_results opt --include-running
python3 -m pruning.plot_results --run outputs/xps/<experiment>/<run>
```

Plotting reads both new result/config schemas and existing Forge runs. Historical static figures and the [supervisor report](outputs/reports/supervisor_progress.md) remain under `outputs/`; their descriptions reflect the implementation at the time they were written. Pre-Forge results were not imported where exact configuration and artifact provenance could not be reconstructed.
