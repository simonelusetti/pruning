# Agent context

## Working preferences

- Work on `main`. Do not create Git branches or commits unless explicitly asked.
  The user authorized the commit containing this handoff only. Algorithm/config
  variants are often called “branches”; that does not authorize a Git branch.
- Use detached tmux for long jobs and verify the worker survives launch. Do not
  repeatedly monitor runs unless requested. Give measured ETAs when possible and
  distinguish them from estimates. Account for stalls, not just per-move time.
- Keep this research framework small. No automated tests by default; narrowly
  check genuinely uncertain mathematics or execution, rather than adding a suite.
- Use PyTorch tensor operations and DataLoader batches. Avoid per-weight loops.
  Algorithms accept an explicit CPU/CUDA device.
- One Forge/Hydra entry point, `train.py`, with reusable helpers. Do not add a
  separate substantial experiment runner for every variant.
- Preserve `sources/`, consult it for published methods, and distinguish those
  papers from project notes and our adaptations. No CNN/Fashion-MNIST additions.
- The user values mathematically clear explanations and dislikes hidden changes
  in masks, sampling, budgets, denominators, or update rules between comparisons.

## Research direction

The shared description is deletion coordinates D, compensating values u on
surviving coordinates U, quadratic Fisher damage, and canonicalized removal K(D):

    1/2 Δ(D,u)^T F Δ(D,u) + μ/2 ||u||²
      + λ_u |supp(u)| - λ_d K(D).

For fixed D and unrestricted U, a positive-definite quadratic has its unique
conditional optimum at `(F_UU + μ I) u = -F_UD d`, with `d = -W_D`.
This does NOT make deletion selection globally optimal or continuous. Ranking
singleton OBS scores and batching the lowest ones is a heuristic for joint D.
An exact fixed-mask quadratic solution also need not minimize true network loss.

The project seeks a cohesive preprint connecting small-model canonicalization
gains and scalable OPT pruning. Historical full-Fisher/canonical MIQP Berman
results were strong; the currently useful scalable OPT reference is global OBS
selection with local full-complement updates. These are related approximations,
not evidence that one identical algorithm already beats every baseline.

## Code and maintained variants

`pruning/runner.py` resolves configs and builds methods/model adapters;
`engine.py` owns proposal, evaluation, acceptance, rollback, and final saving.
`methods/` implements choices; `nnprune/` contains tensor solvers and structural
rules; `models/` implements snapshots, data, and accounting. The root README
documents the detailed config surface.

- `classic`: structural candidates, optionally scalar zeroing.
- `discrete`: generic Fisher MIQP over bounded deletion/update candidates, with
  inferred safe structural reward. Gurobi is required; unrestricted OPT size is
  infeasible and exceeds the installed limited license.
- `continuous`: progressive OPT full-complement quadratic updates. The original
  mask selector uses temporary SparseGPT-style column propagation, which is
  discarded before the actual update solve.
- `continuous_global`: the cleaner continuous preset. Estimate one unchanged-model
  snapshot of local A/B factors per proposal. Rank live weights across ALL
  selected modules by `W_ij² / (2 (B^-1)_ii (A^-1)_jj)`. Select enough additional
  zeros for the cumulative target, then solve each module's full-complement
  quadratic. No per-module quotas, temporary directional propagation, or joint
  deletion-support optimization. Biases fixed. Ordinary runs do not canonicalize;
  some saved endpoints have separate exact post-hoc cleanup artifacts.
- `continuous_neuron`: grouped incoming-row+bias or outgoing-column cuts in the
  fc1/activation/fc2 MLP chain, with canonical removal and local continuous updates.
  Not independent arbitrary row deletions mislabeled as hidden neurons.
- `discrete_simple`: experimental matrix-free canonical L0 alternative, described
  below. It does not replace the original `discrete` backend.
- `discrete_simple_global`: the latest fixed-penalty permanent-deletion preset of
  that L0 method, NOT another OBS method. It is a config, not a Git branch.
- SparseGPT, Wanda, FLAP, and Berman are baselines. Canonical SparseGPT with its
  power-0 control is an older occupancy-bonus experiment, not the current main idea.

## Matrix-free canonical L0

For accepted weights W, optimize final affine values V approximately:

    q(V) + λ R(V),
    q(V) = Σ_m 1/2 tr[B_m (V_m-W_m) A_m (V_m-W_m)^T],
    R(V) = P - K(D(V)).

With canonicalization disabled R is ordinary penalized nonzero count. Eligible
affine matrices may include bias columns. In inferred disjoint MLP pairs the
exact structured proximal map compares ordinary hard thresholding, an incoming
weight-row cut, and an outgoing-column cut for each hidden neuron. These
comparisons are tensor reductions, not candidate-pool enumeration. Constant
hidden outputs are folded into the downstream bias. That protected bias is
exempt from the sparsity charge. The quadratic is evaluated BEFORE exact
function-preserving cleanup; validation evaluates the cleaned network.

Proximal gradient uses `B @ delta @ A`, a power-iteration step estimate and
majorization backtracking, followed by optional fixed-support preconditioned CG.
The default caps are 100 proximal iterations and 30 CG iterations, float32,
relative tolerances 1e-5. These caps frequently prevent convergence; no global
optimum is certified. Diagonal standardization is a change of variables that
preserves the objective, not artificial module-scale normalization.

Each m-by-n product costs O(m²n + mn²); working storage is O(m²+n²+mn) per module,
summed over cached modules. Coupled MLP proximal decisions still use independent
affine Fisher blocks: cross-module curvature is omitted.

Literature: L0Learn section 2.4 motivates penalized IHT/proximal gradient; CHITA
motivates support identification/refinement. The structured canonical proximal
map, K-FAC substitution, and CG refit are project adaptations. This is neither
L0Learn's flagship coordinate/swap algorithm nor a full CHITA reproduction.
Both PDFs are in `sources/`.

Base `discrete_simple` retunes λ for each approximate cumulative target and allows
zeros to regrow. `discrete_simple_global` calibrates one COMMON λ around 5% on
move 1, keeps it fixed, and permanently freezes direct and canonical deletions
out of both search and compensation. Every move refreshes factors around the
last accepted model; there is no quota or increasing target. Trial masks do not
accumulate. Penalty and committed masks live in `problem.extra` for rollback.

The fixed λ does NOT promise another 5% on every move. In practice it plateaued
near 5.7% with tiny positive additions. The current stop rule detects no new
deletions, validation rejection, or protocol limits, but has no minimum-progress
threshold. New random calibration can keep producing tiny moves indefinitely.
The user killed this run. Do not silently increase λ or resume it.

## OPT settings that matter

OPT-125M is decoder-only. “Decoder-linear removal” refers to the selected affine
parameters, not to ignoring a separate encoder. The full model has 125,239,296
unique parameters. The latest L0 universe contains 85,017,600 affine weights and
biases in 72 modules (12 transformer blocks × q/k/v/out/fc1/fc2). Embeddings and
layer norms remain fixed. Other methods can have different eligible counts.

Recent matched-sampling L0/global-OBS runs use 32 FULL 2,048-token sequences for
each of A and B, not 128-token prefixes. B uses model-sampled score gradients,
up to 32×2047 valid token positions per module; A uses input second moments.
Factors are damped by 0.01 times their mean diagonal. Counts, contexts, token
caps, batches, and refresh policies are Forge configs.

Calibration is redrawn once per proposal with seed `sampling.seed + index`, with
A/B sharing prefixes of one sampled pool. Factors stay fixed inside that proposal
and its penalty trials. Recent L0/global OBS refresh both factors once per move;
the older continuous variant can refresh B per transformer block.

Selection uses 4 fixed C4 sequences; independent reporting uses 128 fixed
sequences. These C4 windows come from the first English TRAINING shard, sampled
from documents with replacement using separate seeds. They are not the official
validation split, and separate seeds do not prove document-disjoint splits.
Loss SE is computed across sequence means, not independent-token observations.

Acceptance uses original-dense-relative selection damage <= 5% + 2 paired SE.
Held-out loss is reporting only: an accepted point can exceed 5% held-out damage.
Rejection is recorded and rolled back. Normal `save_model=true` writes the final
accepted checkpoint on graceful completion; an interrupt can leave no model.pt.

## Current evidence and exact run references

All numbers below are percent loss increase, not loss ratios. Removal refers to
the run's own eligible denominator. Resolved configs/results are in the archive;
use those instead of inferring settings from colors or historical graph titles.

| Run | Meaning | Selected evidence |
| --- | --- | --- |
| `e5f30cf6/aba68b3d` | Global OBS continuous, 32 full A/B sequences | Raw 50% removal accepted: held-out +5.09263%; 55% rejected: +8.48685%. A separate canonicalization artifact adjusts its saved endpoint only. |
| `d4b5688e/06151a58` | Canonical L0, retuned cumulative 10pp targets | 40.0933% accepted: +3.35573%; 49.8591% rejected: +9.39231%. |
| `fa8b51a4/c332da0c` | Same canonical L0 with 5pp targets | 40.0178%: +3.48793%; 45.7665% accepted: +6.70026%; 50.5464% rejected: +10.60621%. Smaller steps did not preserve the early advantage. |
| `ab6327ed/d8ad0109` | Same-D, same-F refit accuracy diagnostic | At 49.8591% removal, capped versus converged updates: +9.39231% vs +9.30721%; both rejected. Quadratic damage improved 1.85%, so final-refit truncation alone did not explain the loss gap. |
| `ea9a08cb/e3c34771` | Fixed global λ, permanent deletions | User-stopped after 100 accepted moves, 5.73048% removal, held-out -0.00025915% (effectively unchanged). No model checkpoint. |

The last run used λ=8.370355963904785e-9. It first removed 4.84496%, then plateaued;
late moves took about 6.3 minutes despite removing very little. Its last accepted
point is about 3.8900% of all model parameters. It ran roughly 10.5 hours before
the user stopped it. Forge correctly marks it failed after interruption; the
saved result still has a running status and incomplete proposal after step 100.
Do not call it completed or pretend there is a saved final model.

The saved OPT plot contains its earlier step-82 point, not the step-100 state.
At extremely small loss differences, report uncertainty rather than claiming a
meaningful true-loss improvement. Older baseline victories were not universally
matched in offsets, canonicalization, denominator, and sampling; compare configs
before making preprint claims. The archived supervisor reports may predate
important algorithm changes.

## What remains open

The most recent requested action was migration, not new research or another run.
No experiment should be restarted automatically. The user may next want to decide
how a fixed-λ method can make meaningful progressive moves without the old
cumulative-λ retuning. A minimum-progress stopping rule prevents wasted compute
but does not by itself improve achievable sparsity. Any new schedule is a change
to the method and should be described plainly.

Other unresolved questions include support quality, factor drift/noise, batching
interactions, and earlier accumulated update errors. The accurate final-refit
diagnostic narrows one explanation, not all of them. K-FAC is an approximation to
Fisher; a good conditional quadratic solve is not a guarantee of lower true loss.

On the new machine, first restore artifacts and dependencies, confirm the code
revision and conversation, and inspect the desired config. Avoid launching the
default greedy fixed-penalty run expecting it to finish promptly. Preserve source
archives and existing failed-run metadata; never rewrite provenance to simplify
a graph.
