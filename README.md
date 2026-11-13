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
forge run benchmark=modadd branch=discrete branch.options.fisher_n=32
forge run benchmark=berman branch=discrete
forge run benchmark=network branch=discrete_simple benchmark.model.f=/path/model.pt benchmark.data.f=/path/data.pt
forge run benchmark=opt branch=discrete_simple sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=discrete_simple_fast runtime.device=cpu sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=discrete_simple_global runtime.device=cuda:0 sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=modadd branch=classic runtime.device=cuda:0
forge run benchmark=modadd_berman branch=berman
forge run benchmark=opt branch=continuous
forge run benchmark=opt branch=continuous_global sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=continuous_global_affine runtime.device=cuda:0 sampling.calibration_sequences=32 sampling.b_sequences=32
forge run benchmark=opt branch=continuous_neuron
forge run benchmark=opt branch=sparsegpt
forge run benchmark=opt branch=canonical_sparsegpt sampling.calibration_sequences=32
forge run benchmark=opt_cpu branch=wanda
forge run benchmark=llama_wanda branch=continuous_global_affine_streaming runtime=llama
forge run benchmark=opt_slicegpt branch=native_slicegpt runtime.device=cuda:0
forge run benchmark=llama_wanda branch=native_wanda runtime.device=cuda:0
```

`llama_wanda` and `llama_flap` run our methods on the Wanda and FLAP papers' LLaMA-7B model, calibration data, and WikiText2 evaluation; see [baselines](pruning/baselines/README.md). Language models are cached in `outputs/cache/huggingface/hub`.

Launch long runs inside detached tmux sessions and verify the worker remains active. Work on `main`; leave changes uncommitted unless explicitly instructed otherwise.

The three framework methods are:

- `classic`: structural boundary moves, optionally combined with scalar zeroing using `branch=classic_weights`.
- `discrete`: one model-independent Fisher MIQP for bounded deletion and update pools, with exact canonicalization savings rewarded. Two-moons, modular addition, MNIST, and configured PyTorch models use the same optimizer. Parameter layout and safe linear/activation chains come from the model rather than a fixed architecture or output class count.
- `continuous`: progressive OPT pruning. A blockwise generalized-OBS heuristic selects a deletion mask; temporary columnwise updates used during selection are discarded. The surviving weights in each linear module then receive a full-complement AXB quadratic solve. “Continuous” describes this fixed-mask update, not an exact optimization of the deletion set.

`continuous_global` is a preset of `continuous`. Once per progressive proposal it estimates all selected modules' A and B factors on the unchanged accepted model. It ranks surviving weights across all those modules using `W[i,j]² / (2 * B_inverse[i,i] * A_inverse[j,j])`, then selects exactly the additional deletions needed for the cumulative target. There are no per-block, per-module, or column-chunk quotas, and no temporary SparseGPT propagation. The same local full-complement solver handles compensation, using the snapshot's factors. Biases remain fixed and canonicalization is not applied. Cross-module interactions and joint deletion-set optimization remain approximations.

`continuous_global_affine` preserves that reference and adds an affine variant. Biases join the optimization variables and deletion candidates; A includes a constant input column and is accumulated in float64, as is B. Explicit committed masks fix deletions, while initially or incidentally zero coefficients remain available for compensation. Initially zero coefficients receive no removal credit and are excluded from ranked deletions. The global inverse-Fisher singleton score and five-percentage-point cumulative schedule are retained. One unchanged-model factor snapshot feeds both ranking and converged float64 full-complement solves; the A inverse from ranking is reused for preconditioning. Exact MLP cleanup runs after each proposal without a neuron bonus in the scores. Downstream fc2 biases are protected from deletion because constant folding may change them. Direct and canonical masks participate in rollback and checkpoints, and plotting gives this variant its own curve using the full affine denominator. The unrestricted local inverse score remains a heuristic for a batch under existing fixed deletions; this variant does not claim to solve the joint support problem.

`continuous_neuron` is a preset of `continuous`, with `branch.options.deletion_unit=hidden_neuron`. It considers hidden neurons between `fc1` and `fc2`. Each neuron offers an incoming-row-plus-bias cut or an outgoing-column cut. Grouped compensated scores choose a direction and rank neurons globally by predicted damage per canonical parameter saving. A batch receives the full surviving-MLP quadratic update; function-preserving canonicalization removes the redundant counterpart. Ranking singleton groups into a batch remains heuristic, and cross-module Fisher interactions are omitted.

The supervised `continuous_global_affine` backend also supports Berman/MNIST and configured PyTorch affine models. It uses the same sampled model-Fisher K-FAC estimator as discrete simple, unchanged-model global inverse scores, checked float64 full-complement solves, affine bias compensation, fixed deletion masks, and exact inferred-chain cleanup. Its five-percentage-point schedule is cumulative. Use identical `fisher_n`, `fisher_seed`, and `fisher_batch` settings for a matched curvature comparison. The existing OPT backend retains its estimator and layer replay.

Berman, SparseGPT, Wanda, and FLAP are baselines, selected through the same entry point. Local OPT baseline implementations use the common execution loop; pinned published programs retain their authors' model/data settings and separate environments. Magnitude, OBD-style, global Wanda-style, random and mean-ablation baselines and the four Berman schemes are also composable branch presets. The `classic` method is our neural extension of the supervisor’s boundary-pruning framework. See [baseline definitions and setup](pruning/baselines/README.md).

For another trained model, `benchmark=network` loads a saved `nn.Module` and a tensor dictionary containing `train`, `val`, and `test` pairs `(inputs, targets)`. The validation pair is split into candidate and stopping windows. Model forward returns logits as a tensor or an object with `.logits`. Replace either Hydra `_target_` with your own model or data factory when needed; the optimizer requires no new runner.

```bash
forge run benchmark=network branch=discrete benchmark.model.f=/path/model.pt benchmark.data.f=/path/data.pt
```

Discrete defaults to Bernoulli likelihood for one output per example and categorical likelihood otherwise. Set `branch.options.likelihood=bernoulli` for independent binary outputs, or `gaussian` with `variance` for regression. `fisher_n` (null means all training inputs), `fisher_seed`, and `fisher_batch` control the current-model estimate. Exact Fisher expectations are streamed through three Jacobian passes per proposal: diagonal screening, update correlations, and the restricted quadratic. Neither a dense full Fisher nor the complete example-by-parameter design is retained.

`discrete_simple` is the matrix-free L0 alternative to MIQP. For each proposal's accepted reference weights W, it approximately minimizes `0.5 * (V-W).T F (V-W) + penalty * (P - K(D(V)))`, with independent damped affine K-FAC blocks. Here K counts zeros plus the supported exact structural cleanup, and P is the penalized parameter universe. `canonicalization=false` recovers ordinary L0. An affine matrix includes its bias column by default. All its coefficients are free continuous variables; zeros can regrow within and between proposals. No candidate pool, exact deletion quota, update bound, or update-support penalty is imposed. Other parameter types remain fixed. This backend does not import Gurobi; `discrete` retains the bounded-pool MIQP backend.

The numerical solvers are separate. `solver=fast`, the `discrete_simple` default, restores the historical float32 IHT implementation with power-iteration curvature, at most 100 proximal iterations and one final diagonal-CG refit capped at 30 iterations. Residuals are recorded without requiring convergence. `branch=discrete_simple_fast` selects that solver with approximate cumulative five-percentage-point targets, up to six penalty trials per target, retuned lambda and regrowth, matching the earlier `fa8b51a4/c332da0c` setup. `branch=discrete_simple_accurate` selects the revised adaptive solver and checked float64 inverse-AXB refits. Both use the same structured canonical proximal rule, factors, model adapter, and validation loop; they do not search separately over update support U. Results record `solver_backend`, and plots distinguish progressive accurate runs from fast runs.

The algorithm uses penalized proximal gradient as in [L0Learn, section 2.4](sources/l0learn_fast_best_subset_selection_coordinate_descent.pdf). Ordinary coordinates are thresholded at `sqrt(2 * penalty / Lipschitz)`. For inferred disjoint two-affine chains, our exact structured proximal map compares ordinary thresholding, zeroing a neuron's incoming weights, and zeroing its outgoing weights. An annihilated neuron is charged no retained coefficients, so canonicalization is incentivized during optimization. These comparisons are batched reductions, not a candidate pool. Paired matrices share a step size; their Fisher blocks remain independent. Attention modules retain ordinary L0. Constant outputs are folded into the downstream bias, which is exempt from the L0 charge because folding may change it. The quadratic is evaluated before function-preserving cleanup; validation evaluates the cleaned model. Longer inferred chains retain the coupled canonical count, including cascades and shared middle coefficients counted once. Their structured proximal subproblem uses layer-coordinate descent over ordinary, constant, and disconnected neuron states, rather than claiming an exact global proximal map. The original exact pair rule remains unchanged for disjoint pairs. Chains sharing entire affine modules outside one inferred complete chain remain unsupported.

Curvature products are `B @ delta @ A`. In the accurate solver, proximal steps start from the largest diagonal curvature in the unit, then quadratic-majorization backtracking checks the full AXB direction and canonical objective. The next iteration tries a larger step; before accepting stationarity it retries the initial scale. The diagonal sets a trial step size, without ranking or choosing deletions. Optional support refitting follows [CHITA's identify-and-refine strategy](sources/chita_neural_network_pruning_combinatorial_optimization.pdf), using matrix-free CG rather than its low-rank Woodbury formula. The accurate solver uses inverse-AXB preconditioning and follows provisional stationarity with a support refit and another structured proximal check. The fast solver retains the historical diagonal preconditioner and a single capped refit after IHT. Every proximal iteration revisits the entire affine matrix. The structured proximal map and these AXB solver adaptations are our work, not results claimed by either paper. This is not a reproduction of L0Learn's coordinate/swap solver or CHITA's cardinality-constrained active-set/line-search algorithm.

`standardize=true` applies an exact diagonal change of variables corresponding to unit-norm columns in L0 regression. Both the quadratic and L0 penalty are preserved; module damage scales are not equalized. Cached A/B factors are reused across iterations and penalty trials. `b_refresh=step` refreshes both factors around each accepted stage; `frozen` retains both and requires `sampling.refresh=fixed`. A/B damping changes the curvature metric, not an added ridge penalty on updates. OPT honors the configured modules, block count, full contexts, calibration counts, and B token cap. Other `nn.Module` models use the same factor collector. Legacy functional adapters and tied affine parameters are not supported by this backend.

The default is a progressive curve with approximate cumulative targets of 10%, 20%, and so on up to the configured removal ceiling. Accepted changes are retained; Fisher factors refresh per proposal by default. The first proposal rejected by validation is recorded and rolled back, ending the run. Set `branch.schedule.targets='[0.10]'` for a single-point experiment. `penalty=null` tunes one common penalty across all modules using the requested sparsity only as a hint: a global quantile initializes the penalty, then bounded multiplicative/geometric search uses achieved sparsity. It never selects an exact-size mask. `penalty_trials` and `sparsity_tolerance` control this outer search; an explicit nonnegative `penalty` disables tuning. Targets can be missed because of finite budgets, support jumps, and local minima; results record every trial and whether tolerance was reached. Reporting counts zeros introduced relative to initially nonzero coefficients, without freezing those zeros during optimization. `penalize_bias=false` exempts biases from the L0 charge while retaining continuous bias compensation; name patterns can similarly restrict the penalized universe.

`branch=discrete_simple_global` inherits the accurate solver. It calibrates one model-wide penalty near `initial_sparsity=0.05` on the first proposal, then keeps it fixed. Each subsequent move refreshes AXB and searches all surviving eligible coefficients, with no module quotas or increasing sparsity targets. Explicit deletions and canonicalized neuron parameters are permanently excluded from both the proximal search and compensation. Penalty trials share the same accepted masks; discarded trials do not accumulate deletions. Penalty and masks participate in checkpointing and rollback. Later move sizes are unconstrained, and the path stops on validation rejection, no further deletions, or the usual removal/step limits. The plain `discrete_simple` preset retains its original retuning and regrowth behavior.

The former `discrete_simple_ranked` experiment used a diagonal-Fisher ranked starting mask. That initializer and preset have been removed. Maintained L0 proposals start from the accepted affine values and choose support through the structured proximal objective. Historical ranked-run artifacts and their distinct plot label remain unchanged.

`max_iterations`, `rtol`, and `max_backtracks` bound the proximal search; `refit_iterations=0` explicitly disables support refitting. The accurate solver requires CG to reach `refit_rtol` using a freshly computed residual relative to its fixed-mask right-hand side, or it raises and the proposal is rolled back. Its defaults are float64 arithmetic, up to 2,000 iterations per refit, and relative residual 1e-6; the fast defaults are float32, 30 iterations and 1e-5, with no convergence requirement. Neither a converged refit nor a stationary proximal point certifies the best deletion set. Penalty trials restart from the same accepted reference rather than carrying an overly sparse support into a smaller penalty. No full Fisher or parameter-square matrix is constructed. Only the accurate solver inverts and caches the small A/B factors for preconditioning. With an m-by-n affine matrix, each curvature or inverse-AXB preconditioner product costs O(m²n + mn²), while factors, inverses, and working matrices occupy O(m² + n² + mn) storage. Accurate factor inversion adds O(m³+n³) per module when factors refresh. Full-model memory is the sum of factor caches and linear-size reference, trial, and best-trial buffers. Biases remain continuous compensation variables even when exempt from the sparsity penalty. The shared validation, held-out reporting, rollback, and checkpointing remain in `engine.py`.

`pool` counts globally ranked scalar-deletion blocks and `window` sets their size; `group_pool` adds disjoint outgoing-neuron-column candidates with at most `max_group` live weights each. `update_pool` limits compensating coordinates. Screening remains `0.5 * weight² * Fisher_diagonal`; MIQP evaluates the joint quadratic and structural savings inside this pool. `parameters=null` considers tensors with at least two dimensions; a list of name patterns restricts deletion scope. Biases may compensate, while downstream biases needed for constant folding are protected from explicit deletion. `require_deletion=false` allows a no-op optimum to end the path; the historical Berman preset explicitly requires a deletion and uses 256-weight blocks.

Automatic structural rules use the execution graph, not module declaration order. They cover exclusive, unshared linear/pointwise-activation chains with valid constant folding. Other operations can participate in scalar pruning and Fisher interactions, but receive no invented structural reward. Legacy functional benchmarks provide their chain metadata through the same model interface. `canonicalization=false` disables those rules. Shared masks count explicit and structural removals once and participate in checkpoint/rollback. Historical discrete curves retain their old pool and accounting semantics; a refactor does not establish identical trajectories.

`canonical_sparsegpt` is an experimental SparseGPT-derived method. Once per progressive proposal it multiplies MLP deletion scores by `(remaining_nonzero / group_width) ** completion_power`, using incoming-row occupancy for fc1 and outgoing-column occupancy for fc2. The default power is 1; power 0 disables the bonus for a matched control. Attention uses ordinary SparseGPT scores. This uses A-based scoring and SparseGPT updates, with no B estimation or full-complement solve. Deleted coordinates remain fixed. Exact MLP canonicalization runs after each MLP module; fc2 input statistics are recollected after fc1 pruning and cleanup. Explicit masks count direct and canonical removals once, including removed fc1 biases; changes to fc2 bias from constant folding are not counted as deletions. The power-0 control shares this cleanup, fixed-mask handling, and statistics refresh, so it is distinct from the untouched `sparsegpt` baseline.

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
sources/                 Reference papers, original notes, and archived original code
outputs/                 Runs, caches, checkpoints, figures, historical reports
```

There are no internal experiment subprocesses or per-method argument parsers. Subprocesses are used only to launch the authors' external baseline programs and inspect their pinned environments.

New internal OPT runs count decoder-linear weights and biases in a shared denominator; zero biases already present in the dense model are not credited as pruning. Historical CPU baseline artifacts used a weights-only sparsity field and retain that accounting. Neuron continuous counts structurally removed coordinates rather than incidental numerical zeros.

Each new internal run writes a common versioned `result.json`: `curve` contains evaluated proposals, `runs` includes in-progress method details, `accepted_endpoint` identifies the retained state, and `rejected_candidate` preserves the first failed validation proposal. `model.pt`, when requested, contains `state`, `target_sparsity`, and `removed_percent`; a PyTorch adapter's state contains `parameters` and `structure`, while local adapters retain their `State` object. Full configs and metrics are managed by Forge.

Forge's `on_finish` command automatically refreshes `outputs/plots/<benchmark>/comparison.png` after a successful run. The default `[python3, pruning/plot_results.py]` is resolved by `train.py` to the current interpreter and an absolute script path before Forge changes directories. Set `forge.on_finish=null` to disable it, or supply a different command argument list. The command is excluded from experiment identity; plotting failure does not change a run's outcome.

Every method runs until it breaks. Progressive and greedy branches prune the accepted model further at each step; branches with a fixed target, including the authors' programs, are swept in five-percentage-point steps, each applied to the dense model (`independent`). Both stop at the first step whose validation loss exceeds the original by more than `rho` (5%) plus `n_sigma` (2) paired standard errors, and keep the last accepted state.

After stopping, the engine drops fully dead units from that accepted state (an MLP unit whose output weights are zero, or whose input weights and bias are zero before an activation with f(0)=0) and times the held-out evaluation. The same measurement taken on the dense model at the start of the run gives the reference. `result.json` records both under `inference`, with the parameter bytes of the compacted model and the held-out loss change caused by compaction (zero up to floating-point reordering). Zeros scattered inside a matrix stay in dense storage, so they change neither measurement.

[`pruning/plots.py`](pruning/plots.py) draws one four-panel figure per benchmark from the latest completed run of each method, where each branch config (its `label`) is its own method: held-out loss increase against the removed share of the method's considered parameters and of all unique model parameters (shared weights counted once), then inference time and parameter memory at the last accepted state as a percentage of the same run's dense measurement. Only saved `result.json` curves are drawn; runs without inference measurements are left out of the bar panels. First rejected steps are marked with ×. Concurrent hooks serialize plot updates and replace images atomically.

```bash
forge grid benchmark=moons --run branch=classic --run branch=classic_weights --run branch=discrete
forge metrics benchmark.name=moons
forge artifact 'result.json'
python3 -m pruning.plot_results opt
python3 -m pruning.plot_results opt --include-running
python3 -m pruning.plot_results --run outputs/xps/<experiment>/<run>
```

Plotting reads the current `curve` result schema; older `history` results are not drawn. Historical static figures and the [supervisor report](outputs/reports/supervisor_progress.md) remain under `outputs/`; their descriptions reflect the implementation at the time they were written. Pre-Forge results were not imported where exact configuration and artifact provenance could not be reconstructed.
