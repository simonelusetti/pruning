# Baselines

All baseline experiments enter through `train.py` and Forge. `berman.py`, `language.py`, and `modadd.py` provide algorithms to the shared execution loop. `sparsegpt.py` is the matrix reconstruction primitive.

A baseline is a published method together with the model and data it was published on. Each paper's benchmark reproduces its model, calibration data, and evaluation; our methods and the authors' own program (`native_*` branches) both run on it:

| Paper | Model, calibration, evaluation | Native `branch=` | `benchmark=` |
| --- | --- | --- | --- |
| SparseGPT | OPT-125M; C4 128×2048 | none: our `sparsegpt` re-implementation is the baseline | `opt` |
| Wanda | LLaMA-7B; C4 128×2048; WikiText2 test, 2048-token windows | `native_wanda` | `llama_wanda` |
| FLAP | LLaMA-7B; WikiText2 1024×128; WikiText2 test, 128-token windows | `native_flap` | `llama_flap` |
| Berman Fisher distance | SimpleNN; MNIST | `berman_one_shot`, `berman_iterative`, `berman_global`, `berman_exact` | `berman` |
| SliceGPT | OPT-125M; WikiText2 128×2048; WikiText2 test, 2048-token windows | `native_slicegpt` | `opt_slicegpt` |
| LLM-Pruner | LLaMA-7B; BookCorpus 10×64; WikiText2 test, 128-token windows; block-wise Taylor on layers 4–29 | `native_llm_pruner` | `llama_llm_pruner` |
| Kwon et al. 2022 | Fine-tuned BERT-base / DistilBERT; GLUE and SQuAD | not available: the authors' fine-tuned checkpoints (Google Drive) return 404 | — |

SparseGPT, Wanda, FLAP, SliceGPT, LLM-Pruner, and Kwon et al. are the stronger baselines listed in the supervisor's archived draft (`sources/original code/paper/main.tex`, outlook item 3); its shared short fine-tuning budget is not implemented. The LLaMA papers' `decapoda-research/llama-7b-hf` checkpoint has been removed from Hugging Face. All LLaMA presets use `huggyllama/llama-7b`, a mirror of the same LLaMA-7B weights; `specs.py` and the benchmark configs record the paper checkpoint as `paper_model`. Each matched benchmark selects its paper's sampling preset (`wanda`, `flap`, `slicegpt`, `llm_pruner`), so their contexts, calibration sets, and full WikiText2 test windows follow the native evaluation. Our held-out loss is the mean of equal-length window losses; its exponential is the perplexity the native programs print. Our runs additionally need a validation split for the stop rule; `llama_llm_pruner` uses WikiText2 validation windows because BookCorpus has no held-out split. BookCorpus calibration reads the same archive as the hub loading script, which `datasets` 4 no longer executes. Native outputs are scored in the benchmark's dtype, like every other method.

The authors' [Wanda example](https://github.com/locuslab/wanda#usage) uses LLaMA-7B with 50% unstructured sparsity. The [FLAP example](https://github.com/CASIA-LMC-Lab/FLAP#configuration-instruction) uses LLaMA-7B with about 20% structured removal, WIFV, AL-AM, and 1,024 calibration examples. These are external methods to compare against our `classic`, `joint`, and `singular` branches. Running all methods on one identical checkpoint, data protocol, and removal budget is a separate controlled comparison from reproducing each paper's native setup. Current internal `wanda` and `flap` presets use OPT; `reference` with `score=wanda scope=global` is the notebook's global-ranking control and has a different selection rule from published row-wise Wanda.

`classic` is our neural extension of the supervisor's boundary-pruning framework. The SR paper's `zero_only`, `sloppy_only`, and `bic_only` operators are not separate neural competitors in this repository.


The supervisor's [NNpruning notes](../../sources/NNpruning%20%E2%80%94%20working%20notes.pdf), archived [notebook](../../sources/original%20code/NNpruning.ipynb), and [core](../../sources/original%20code/nnprune/core.py) define the ranking controls and the Berman schemes below.

| `branch=` | Behaviour |
| --- | --- |
| `reference` | Uncompensated ranking: `options.score` magnitude (default), `wanda` (`abs(W) · norm(x)`) or `random`; `scope=global` ranks all pruned tensors together, `local` removes the target fraction from each tensor |
| `berman_one_shot` | Frozen `sqrt(F(theta)) * abs(theta)` ranking; independent points |
| `berman_iterative` | Recompute that score on the accepted model each step |
| `berman_global` | `abs(theta) * mean_alpha sqrt(F(alpha * theta))`, scaling every parameter |
| `berman_exact` | `abs(theta) * sqrt(mean_alpha F_kk(theta with only coordinate k scaled))` |
| `berman_magnitude` | Author-style magnitude control, all eligible parameters, independent points |
| `berman_fim` | Author-style Fisher-diagonal-only control, independent points |

Ranked baselines use no compensation or retraining. `branch.options.refresh=step` with `branch.schedule.mode=progressive` selects an iterative ranking. Biases are excluded by default; `include_bias=true` includes them for PyTorch models. Local adapter pruning scopes follow the archive. Structural cascades for local ranked/Berman reporting are evaluated on a copy and do not alter the running baseline.

Berman's path schemes use the vendored Fisher routines and the archived formulas, three quadrature points, and a 500-example training subset with seed 42 by default. The new Berman presets protect the modular-addition positional embedding, following the archived comparison. The older `branch=berman` retains its historical scope. `avg_points`, `probe_batch`, `sample_chunk`, subset size, and batching remain config settings. The coordinate-path scheme probes every active coefficient and can be expensive; the archive did not run it on modular addition. No new baseline experiment is launched merely by composing a config.

```bash
forge run benchmark=berman branch=joint_simple runtime.device=cuda:0
forge run benchmark=berman branch=singular runtime.device=cuda:0
forge run benchmark=moons branch=reference
forge run benchmark=modadd branch=berman_global
```

For the authors' pinned programs, bootstrap their repositories:

```bash
python3 -m pruning.baselines.bootstrap
forge run benchmark=opt_slicegpt branch=native_slicegpt runtime.device=cuda:0
```

`specs.py` pins each program's commit, environment, dependency versions, and published model and calibration settings; a `native_*` branch refuses a benchmark that does not reproduce them. Repositories live in `outputs/vendor/native_baselines/` and environments in `outputs/envs/`: `published_llm` (Wanda, FLAP; Python 3.9, plus `safetensors` 0.3.3 so the pinned Transformers loads the mirror's safetensors weights), `slicegpt` (Python 3.11; its `pyproject.toml` pins transformers 4.41.0 and datasets 2.18.0 and requires torch ≥ 2.10, which resolved to 2.15; its script also imports `wandb`), and `llm_pruner` (unpinned upstream; Python 3.9 with torch 1.13.1, transformers 4.28.1, datasets 2.11.0, and accelerate 0.18.0, contemporary with its code). `runtime.python` overrides the interpreter. Native programs require CUDA.

`native.py` makes each program a method of the shared engine. A proposal runs the program at the next target from the dense model and saves its output; [`native_eval.py`](native_eval.py) loads that output in the program's own environment and scores it on the benchmark's validation and held-out windows, so stopping, held-out loss, parameter accounting, and inference measurement follow the same protocol as our methods. Wanda and FLAP outputs are rebuilt with each saved layer's shape (FLAP saves smaller layers and compensation biases); LLM-Pruner reloads its own pickled model and SliceGPT its sliced checkpoint. The considered parameters are the nonzero coefficients inside decoder layers, norms excluded; the total counts every unique parameter, so SliceGPT's untied OPT head shows up as added parameters. Each step's console output and scores stay in the run's `native/` directory; the program's own perplexity is kept as `native_reported_ppl` for cross-checking, and the saved model is deleted after scoring. Wanda, FLAP, and LLM-Pruner load [`compat/sitecustomize.py`](compat/sitecustomize.py): current `allenai/c4` metadata makes the pinned `datasets==2.11` reject single-file loads by split count, and the shim disables only that verification. The LLaMA programs read the model from the framework's shared `outputs/cache/huggingface/hub`.
