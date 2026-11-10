# NNpruning: boundary pruning for neural networks

This is a port of the emergentSR pruning idea (local geometry plus boundaries of
parameter space) from symbolic models to neural networks. There are two test
beds:

- **The moons MLP** from `pruningNN/pruningNN.ipynb`: 2-50-50-50-1, sigmoid,
  Bernoulli likelihood.
- **A one-layer transformer** trained from scratch on `a + b (mod 59)`: 4 heads,
  d = 64, d_head = 16, ReLU MLP with 256 units, no LayerNorm.

```
nnprune/core.py         states, moves, paired-2σ selection rule, greedy path, baselines
nnprune/mlp.py          sigmoid-MLP adapter (moves, folds, data, training)
nnprune/transformer.py  transformer adapter (head / attention / rank / unit moves)
nnprune/spectrum.py     gap classifier (port of criteria_bench) + block Fisher diagnostics
nnprune/fdist_baseline.py  Berman et al. (2026) Fisher-distance schemes as baselines
nnprune/third_party/    their Fisher code, vendored verbatim (Apache-2.0, license included)
nnprune/experiments.py  experiment drivers with CSV caching
run_experiments.py      command line:  python run_experiments.py {mlp,mlp-spectrum,transformer,fdist-mlp,fdist-transformer}
NNpruning.ipynb         driver notebook: runs (or loads) everything and plots
fair_comparison.ipynb   granularity x selection-data design (built by build_fair_notebook.py)
tests/test_core.py      python -m pytest -q tests
tests/test_fdist.py     faithfulness of the Berman baseline (needs their repo + nngeometry)
results/                cached CSV / JSON (and the trained base states)
```

Requirements: torch ≥ 2.0 (it uses `torch.func`), numpy, pandas, scikit-learn,
and matplotlib.

## The procedure

**1. Moves go to boundaries, and limits are stored by their exact representative.**
No limit is ever stored as a large number.

| move | limit | realised as | freed (k_eff) |
|---|---|---|---|
| `zero` | weight → 0 | mask | 1 per weight, plus cascades |
| `unit_sat` | sigmoid: b → ±∞, or any incoming weight → ±∞ when the inputs are positive (layers ≥ 1); ReLU: b → −∞ | constant 0/1 folded into the next bias | fan-in + bias + fan-out |
| `unit_mean` | whole fan-in → 0 | the unit's mean, folded into the next bias | same |
| `unit_step` | layer-0 sigmoid, w_ij → ±∞ with sign-varying input | Heaviside unit H(s·x_j) | fan-in + bias |
| `unit_merge` | duplicate-neuron singularity, a_i ≈ α a_i' + β | folded into unit i' and the next bias | unit i |
| `head_off` / `head_mean` | head output → 0 / → its mean | mask / per-position bias `B_attn` | Q, K, V, O of the head |
| `attn_uniform` | softmax scale s → 0 | W_Q, W_K masked | Q, K |
| `attn_fixed` | pattern constant in the input (includes hard-max with a fixed argmax) | fixed pattern | Q, K minus T(T−1)/2 |
| `qk_rank` / `ov_rank` | rank of W_Q W_Kᵀ (or W_V W_O) minus one | activation-weighted SVD, closed form | 2d per rank |

**2. Geometry proposes.** Weight moves are ranked by the second-order predicted
cost $-g_j\theta_j + \tfrac12 F_{jj}\theta_j^2$, using the diagonal GGN (OBD with
the gradient term kept). The few hundred structural moves are ranked by the
exact change in training loss on a subset. The top candidates of each kind go
to the validation test.

**3. Data decides.** All tests below are on the validation split and use the
per-sample paired difference:

- **Anchored admissibility.** A move is admissible if
  $\overline{c(f_m)-c(f_0)} \le n_\sigma\,\widehat{SE}$ (+ `tol_abs`), where
  $f_0$ is the *original* model, not the previous step.
- **Tie set.** The admissible moves within $n_\sigma$ paired SE of the best admissible move.
- **Tie-break** (`tie_break`):
  - `k_eff_then_layer` (default): most parameters removed first, then the
    earliest layer.
  - `k_eff`: most parameters removed.
  - `layer`: earliest layer.
  - `none`: lowest validation criterion.
- **Criterion** (`criterion`): per-sample log loss (`loss`) or 0/1 error (`error`).

The path stops when no proposed move is admissible, after one retry with four
times more candidates. The test split is never used for a decision.

`k_eff` is the number of live parameters after the structural cascades. It is
the k of emergentSR's BIC tie-break, with induced nulls counted. Continuous
gauges (ReLU rescaling, the GL(d_h) symmetry of QK and OV) are not subtracted;
`spectrum.py` measures them instead.

## Baselines

- `magnitude`: |w|.
- `obd`: $-g w + F w^2/2$.
- `wanda`: |w| times the input-activation norm. This uses global ranking, not
  Wanda's per-row grouping.
- `random`.

Each of these zeroes the lowest-scored fraction in one shot, with no
compensation or retraining, and is evaluated on a sparsity grid.

- `mean_ablation`: the classical structured baseline. Units (and heads) are
  removed in order of the variance of their contribution, smallest first, and
  each is replaced by its mean.

- `fdist_one_shot`, `fdist_iterative`, `fdist_global`, `fdist_exact`: the
  Fisher-distance family of Berman, Fu, Hirst & Obirai (2026,
  arXiv:2609.16129).
  - **Their code:** `calculate_fim_backprop` and the forward-mode kernel
    `fisher_entries_forward` are vendored verbatim from
    https://github.com/edhirst/Fdist_Pruning (commit `8d60ea9`, Apache-2.0).
  - **Their protocol:** model Fisher on a 500-point training subset (seed 42),
    K = 3, global unstructured pruning, no retraining. All parameters are
    eligible for the MLP; for the transformer, all but the positional
    embedding.
  - **Faithfulness check:** `tests/test_fdist.py` verifies that our runs zero
    exactly the entries their pruner classes zero. To run it:
    `git clone https://github.com/edhirst/Fdist_Pruning`, `pip install nngeometry`,
    then `FDIST_REPO=/path/to/Fdist_Pruning python -m pytest -q tests/test_fdist.py`.
  - The exact scheme is run on the MLP only, because on the transformer it
    needs about 57k × K forward-mode probes per sparsity step.

Every baseline is evaluated with the same anchored test, cut at its first
failure, and credited in k_eff units with the same cascades.

## What is classical and what is not

**Classical:**

- OBD saliency (LeCun et al. 1990).
- Mean-replacement and bias-compensation neuron removal (e.g. Castellano et al.
  1997).
- Neuron merging of near-duplicates (Srinivas & Babu 2015).
- Head mean-ablation.
- Activation-aware low-rank factorisation (ASVD / SVD-LLM style).
- The one-standard-error rule.
- Magnitude, OBD and Wanda as baselines.
- The Fisher–Rao length of the coordinate path to zero, $\sqrt{I_{kk}}\,|\theta_k|$
  (Berman et al. 2026). In a singular model this is the length of one path,
  not the distance to the pruned submodel.

**Specific to this framework:**

- Treating every one of these as a boundary of a compactified parameter space,
  realised by its exact limit. This adds the saturation and Heaviside
  endpoints, which only exist because of the boundary view.
- One selection rule over all move types: anchored paired-n_σ admissibility,
  paired tie set, k_eff/layer tie-break.
- k_eff with cascades as the complexity count.
- The tolerance-free gap test applied to network blocks, to separate exact
  structural nulls from sloppiness.

## Fair comparison (`fair_comparison.ipynb`, `run_experiments.py fair-mlp|fair-transformer`)

This design separates where the gain comes from. Two factors are crossed, and
Berman et al.'s iterative score is the single-parameter proposer everywhere:

- **Granularity:** W (single parameters), S (structural moves), W+S.
- **Selection data:** one-shot (static), train-selected, valA-selected.

All methods share one stop rule: the anchored paired 2σ test on **valB**, a
half of the validation split that no method uses to choose anything, cut at the
first failure. The select split only ranks candidates and forms tie sets, so
selecting on a larger split cannot end a path earlier through a more powerful
test. Tie-break: fewest remaining parameters. `stop_points()` recomputes the
cut for a practical margin δ; see `fair_findings.md` for why δ = 0 is brittle.

## Flags (`PruneConfig`)

`n_sigma` (2), `tol_abs` (0), `tie_break`, `criterion`, `weight_chunk`,
`weight_per_site`, `use_struct`, `struct_kinds`, `n_struct`, `n_merge`,
`refresh_every`, `screen_n`, `n_mc`, `max_steps`, `expand_on_fail`, `seed`.

Added for the fair comparison. With the defaults the original behaviour is
unchanged, bit for bit:

- `select_split`: which split chooses each move (`val`, `valA` or `train`).
- `stop_split`: the split of the anchored stop test.
- `stop_mode`: `halt` ends the path at the first failure; `mark` records the
  test and keeps going.
- `select_admissibility`: whether the anchored test on the select split also
  filters candidates.
- `post_stop_steps`: in `mark` mode, how many steps to keep after the first
  failure.
- `max_frac`: end the path once this fraction of k_eff is removed.
- `weight_score`: the proposer for single-parameter moves (`obd`, `magnitude`,
  or `fdist`, Berman et al.'s iterative score computed with their code).

`weight_chunk` is a computational device (the transformer uses 64), but it
interacts with the k_eff tie-break: a chunk competes as one move removing
`weight_chunk` parameters.
