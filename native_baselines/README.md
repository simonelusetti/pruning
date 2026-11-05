# Baseline support files

Baseline experiments run through Forge from the repository root. See the root README and `conf/branch/` for Berman, local OPT SparseGPT/Wanda/FLAP, and pinned published settings.

`bootstrap.py` checks out the published SparseGPT, Wanda, and FLAP repositories at the commits in `specs.py`. `sparsegpt.py` contains the local matrix pruning primitive. The benchmark implementations are in `pruning_experiments/`; Forge stores their results under `outputs/xps/`.

Published implementations require their own environments and CUDA. Select a matching pair such as `benchmark=published_sparsegpt branch=sparsegpt_published`, set `runtime.device=cuda:N`, and set `branch.options.python` to the corresponding environment's Python executable. The Wanda and FLAP presets use their published 7B LLaMA model and data settings.
