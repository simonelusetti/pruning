# Pruning experiments

`train.py` is the single experiment entry point, called by Forge. Hydra configs select the benchmark, method, sampling, and evaluation protocol. Internal methods run through direct Python calls and one shared execution loop. Forge owns configuration identity, run status, scalar logs, metrics, and artifacts under `outputs/xps/`.

Reference papers, original notes, and the archived original implementation live in [`sources/`](sources/). Consult this archive when working with published methods; its original code is reference material rather than part of the active framework.

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

CPU and CUDA use the same tensor algorithms. Select the device with `runtime.device=cpu` or `runtime.device=cuda:0`; CUDA requires a CUDA-enabled build of the pinned PyTorch version and a working NVIDIA driver. Tokenization, reporting, and some layer-replay buffers remain on CPU. Gurobi's combinatorial solve remains on CPU even when Fisher estimation uses CUDA.

## Running

```bash
forge run
forge run benchmark=moons branch=discrete runtime.device=cpu
forge run benchmark=berman branch=continuous runtime.device=cpu
forge run benchmark=berman branch=discrete_simple branch.scope=local branch.unit=weight
forge run benchmark=berman branch=discrete branch.curvature=kfac
forge run benchmark=opt branch=continuous sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=discrete_simple branch.scope=local branch.replay=true
forge run benchmark=opt branch=discrete_simple_global runtime.device=cuda:0
forge run benchmark=opt branch=sparsegpt
forge run benchmark=opt branch=reference branch.options.score=wanda branch.scope=local
forge run benchmark=modadd branch=classic runtime.device=cuda:0
forge run benchmark=modadd_berman branch=berman
forge run benchmark=network branch=discrete_simple benchmark.model.f=/path/model.pt benchmark.data.f=/path/data.pt
forge run benchmark=llama_wanda branch=continuous branch.factor_backend=streaming runtime=llama
forge run benchmark=opt_slicegpt branch=native_slicegpt runtime.device=cuda:0
```

`llama_wanda` and `llama_flap` run our methods on the Wanda and FLAP papers' LLaMA-7B model, calibration data, and WikiText2 evaluation; see [baselines](pruning/baselines/README.md). Language models are cached in `outputs/cache/huggingface/hub`.

Launch long runs inside detached tmux sessions and verify the worker remains active. Work on `main`; leave changes uncommitted unless explicitly instructed otherwise.

## Methods

Our curvature methods are one branch each; their variants are shared settings ([`conf/settings/default.yaml`](conf/settings/default.yaml)) that mean the same thing in every method:

| Setting | Values | Meaning |
| --- | --- | --- |
| `scope` | `global`, `local` | `global`: one model-wide budget and problem. `local`: each solve unit removes the target fraction of its own parameters. |
| `replay` | `false`, `true` | With `scope=local` and K-FAC: units are pruned block by block, each measured on the already pruned model (A and B re-estimated per block). |
| `unit` | `both`, `weight`, `neuron` | Deletion candidates: single coefficients, whole hidden units, or either with neuron savings credited. |
| `affine` | `true`, `false` | Biases are deletion candidates and compensation variables, or frozen. |
| `curvature` | `kfac`, `fisher` | Block-diagonal K-FAC (no cross-module terms) or the exact model Fisher (cross-module terms; small models only). |
| `factor_backend` | `memory`, `streaming` | Streaming estimates K-FAC factors one decoder block at a time on the unchanged model and keeps factors and scores in `runtime.scratch`. |
| `estimation.*` | | Fisher sample (`n`, `seed`, `batch`), likelihood, variance and A/B damping, shared by every method. |

A solve unit is one affine module, or a whole canonical group (an MLP chain or gated LLaMA MLP) when neurons are candidates. Under K-FAC a global problem separates into per-unit problems linked by one shared price per deletion, so `global` differs from `local` only in how the budget is shared. Function-preserving cleanup (dead neurons, folded constants) runs after every step in every method, and accounting counts initially nonzero coefficients that are zero. Plot labels are built from these settings, e.g. `continuous · global · both · affine · kfac`.

| Branch | Curvature | Supported settings |
| --- | --- | --- |
| `continuous` | K-FAC | all scopes, units, affine, replay, backends |
| `discrete_simple` | K-FAC (default) or exact Fisher | all scopes, units, affine; replay and streaming with K-FAC |
| `discrete` | exact Fisher (default) or K-FAC | `scope=global`; all units and affine |

- `continuous` ranks candidates by singleton K-FAC OBS cost per removed parameter: `0.5 w² / (B⁻¹ᵢᵢ A⁻¹ⱼⱼ)` for a weight, the cheaper of an incoming-row cut (`0.5 wᵢ A wᵢ' / B⁻¹ᵢᵢ`) or outgoing-column cut for a neuron. It selects the cheapest candidates up to the budget, then solves every surviving coefficient of each changed module in one converged, preconditioned CG (the full AXB complement). Deletions stay fixed. Selection ignores interactions between the chosen candidates; the compensation is exact under K-FAC.
- `discrete_simple` approximately minimizes `0.5 (V-W)' F (V-W) + λ · retained(V)` by penalized proximal gradient ([L0Learn, section 2.4](sources/l0learn_fast_best_subset_selection_coordinate_descent.pdf)), where `retained` counts nonzeros after canonical cleanup. With neuron candidates, the proximal map of each canonical unit also compares zeroing a neuron's incoming weights (a folded constant) and its outgoing weights, charging an annihilated neuron nothing; `unit=neuron` keeps single coefficients and only makes neuron decisions. λ is searched geometrically toward each target, over all units at once (`global`) or per unit (`local`); zeros may regrow. `solver=fast` (default) uses a power-iteration step, float32 and one capped diagonal-preconditioned refit; `solver=accurate` adapts its step, retries before accepting stationarity, and requires its refits ([CHITA's identify-and-refine](sources/chita_neural_network_pruning_combinatorial_optimization.pdf)) to converge, so use it with `refit_iterations=2000 refit_rtol=1e-6 solver_dtype=float64`. `standardize=true` is the exact unit-diagonal change of variables.
- `discrete` solves a Fisher MIQP with Gurobi over a small pool: `pool` blocks of `window` coefficients screened by `0.5 θ² F_ii`, plus `group_pool` outgoing neuron columns (`unit=neuron` uses only columns, `unit=weight` only blocks and no structural reward). `update_pool` coordinates chosen by Fisher correlation compensate. Gurobi's restricted pip licence covers these pools.

Three branches keep their own behaviour:

- `classic` is our neural extension of the supervisor's boundary-pruning framework (`classic_weights` adds scalar zeroing).
- `discrete_simple_global` calibrates one global λ near `initial_sparsity` on its first greedy move, then keeps it fixed with permanent deletions until no coefficient leaves.
- Baselines: Berman's six schemes, our re-implementations of SparseGPT, Wanda and FLAP, the authors' own programs (`native_*`), and `reference`, an uncompensated ranking control with `options.score` magnitude, wanda (`|W|·‖x‖`) or random and its own `scope`. See [baseline definitions and setup](pruning/baselines/README.md).

For another trained model, `benchmark=network` loads a saved `nn.Module` and a tensor dictionary containing `train`, `val`, and `test` pairs `(inputs, targets)`. The validation pair is split into candidate and stopping windows. Model forward returns logits as a tensor or an object with `.logits`. Canonical chains are inferred from the execution graph (exclusive linear/pointwise-activation chains with valid constant folding). `estimation.likelihood` defaults to Bernoulli for one output and categorical otherwise; set `gaussian` with `variance` for regression.

## Configuration

| Config | Responsibility |
| --- | --- |
| `benchmark.*` | Model/data definition, split sizes, seeds, model-specific method defaults |
| `branch.*` | Shared method settings (`scope`, `unit`, ...); `branch.options.*` holds algorithm settings, overriding `benchmark.methods.<method>` |
| `branch.schedule.*` | Greedy, progressive, or independent runs; targets and spacing |
| `sampling.*` | Language-model calibration and evaluation sequence counts, contexts, seeds |
| `protocol.*` | Candidate selection, validation stopping, reporting cadence, final model saving |
| `runtime.*` | Device, CPU threads, external baseline interpreter; excluded from experiment identity |

`branch.schedule.mode=progressive` retains accepted changes. `independent` restores the dense model before each sparsity point. `greedy` proposes one candidate move at a time. All use the same evaluation, acceptance, rollback, and recording implementation.

For OPT, `branch.options.modules` restricts the linear modules being pruned. Optional `branch.options.accounting_modules` selects the percentage denominator and must include those modules; it defaults to the pruning scope. For example, `modules="fc1,fc2"` with `accounting_modules=all` prunes only MLPs but measures removal against all transformer-block linear parameters.

For example, tune continuous sampling without editing code:

```bash
forge run benchmark=opt branch=continuous sampling.calibration_sequences=16 sampling.b_sequences=16 sampling.heldout_sequences=32
forge run benchmark=opt branch=continuous branch.schedule.targets='[0.05,0.10,0.15]' protocol.save_model=true
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

`sampling.refresh=step` redraws calibration once per pruning proposal, with seed `sampling.seed + proposal_index`; `fixed` reuses the initial draw. Our methods default to `step`; classic and baseline presets default to `fixed`. Without replay, K-FAC factors come from one snapshot of the accepted model per proposal. Selection and reporting windows remain fixed throughout a run.

C4 sampling uses the first English training shard, chooses documents with replacement, rejects short documents, and draws contiguous windows. The shard loads once per process. Separate seeds do not guarantee document-disjoint samples; current C4 reporting uses training-shard windows, not the official validation split. Classic candidate and acceptance sets use separate windows. Every result records its actual split metadata and calibration seeds. Historical results retain their original settings.

`protocol.selection_check=true` accepts a proposal when its paired validation loss increase is at most `protocol.rho` times the original loss plus `protocol.n_sigma` standard errors. Rejection restores the accepted model. `false` skips this check and its data loading on OPT; no true-loss stopping guarantee remains. Classic candidate comparison is a separate choice: `protocol.candidate_selection=loss` or `proxy`. Reporting never chooses or accepts proposals.

Reporting runs every `protocol.reporting_every` proposals, default 1; 0 reports only the dense model and final accepted endpoint. The historical `opt_classic` preset uses endpoint reporting to avoid a full test evaluation for every small move. Standard errors use sequence means. `protocol.save_model=true` saves the final accepted state, including structural masks.

## Code layout and artifacts

```text
train.py                 Hydra + Forge lifecycle
pruning/runner.py        Resolve configs; construct model adapter and method
pruning/engine.py        Shared proposal/evaluation/acceptance/rollback loop
pruning/models/          Model loading, state snapshots, split evaluation, accounting
pruning/methods/         Our methods; kfac.py holds what they share (settings, units, cleanup, curvature)
pruning/nnprune/         Tensor solvers, canonicalization, local model primitives
pruning/data.py           Shared language-data sampling
pruning/evaluation.py     Losses and paired statistics
pruning/results.py        Forge logs, result.json, optional model.pt
pruning/baselines/        Baseline algorithms and pinned external integrations
conf/                    All experiment presets
sources/                 Reference papers, original notes, and archived original code
outputs/                 Runs, caches, checkpoints, figures, historical reports
```

There are no internal experiment subprocesses or per-method argument parsers. Subprocesses are used only to launch the authors' external baseline programs and inspect their pinned environments.

New internal OPT runs count decoder-linear weights and biases in a shared denominator; zero biases already present in the dense model are not credited as pruning. Historical CPU baseline artifacts used a weights-only sparsity field and retain that accounting.

Each new internal run writes a common versioned `result.json`: `curve` contains evaluated proposals, `runs` includes in-progress method details, `accepted_endpoint` identifies the retained state, and `rejected_candidate` preserves the first failed validation proposal. `model.pt`, when requested, contains `state`, `target_sparsity`, and `removed_percent`; a PyTorch adapter's state contains `parameters` and `structure`, while local adapters retain their `State` object. Full configs and metrics are managed by Forge.

Forge's `on_finish` command automatically refreshes `outputs/plots/<benchmark>/comparison.png` after a successful run. The default `[python3, pruning/plot_results.py]` is resolved by `train.py` to the current interpreter and an absolute script path before Forge changes directories. Set `forge.on_finish=null` to disable it, or supply a different command argument list. The command is excluded from experiment identity; plotting failure does not change a run's outcome.

Every method runs until it breaks. Progressive and greedy branches prune the accepted model further at each step; branches with a fixed target, including the authors' programs, are swept in five-percentage-point steps, each applied to the dense model (`independent`). Both stop at the first step whose validation loss exceeds the original by more than `rho` (5%) plus `n_sigma` (2) paired standard errors, and keep the last accepted state.

After stopping, the engine drops fully dead units from that accepted state (an MLP unit whose output weights are zero, or whose input weights and bias are zero before an activation with f(0)=0) and times the held-out evaluation. The same measurement taken on the dense model at the start of the run gives the reference. `result.json` records both under `inference`, with the parameter bytes of the compacted model and the held-out loss change caused by compaction (zero up to floating-point reordering). Zeros scattered inside a matrix stay in dense storage, so they change neither measurement.

[`pruning/plots.py`](pruning/plots.py) draws one four-panel figure per benchmark from the latest completed run of each method, where each method label (branch plus settings) is its own method: held-out loss increase against the removed share of the method's considered parameters and of all unique model parameters (shared weights counted once), then inference time and parameter memory at the last accepted state as a percentage of the same run's dense measurement. Only saved `result.json` curves are drawn; runs without inference measurements are left out of the bar panels. First rejected steps are marked with ×. Concurrent hooks serialize plot updates and replace images atomically.

```bash
forge grid benchmark=moons --run branch=classic --run branch=classic_weights --run branch=discrete
forge metrics benchmark.name=moons
forge artifact 'result.json'
python3 -m pruning.plot_results opt
python3 -m pruning.plot_results opt --include-running
python3 -m pruning.plot_results --run outputs/xps/<experiment>/<run>
```

Plotting reads the current `curve` result schema; older `history` results are not drawn. Historical static figures and the [supervisor report](outputs/reports/supervisor_progress.md) remain under `outputs/`; their descriptions reflect the implementation at the time they were written. Pre-Forge results were not imported where exact configuration and artifact provenance could not be reconstructed.
