# Baselines

All baseline experiments enter through `train.py` and Forge. `berman.py`, `language.py`, and `modadd.py` provide algorithms to the shared execution loop. `sparsegpt.py` is the matrix reconstruction primitive.

For the authors' pinned programs, bootstrap their repositories:

```bash
python3 -m pruning.baselines.bootstrap
forge run benchmark=published_sparsegpt branch=sparsegpt_published runtime.device=cuda:0 runtime.python=/path/to/sparsegpt-env/bin/python
```

`specs.py` pins commits, model/data settings, and dependency versions. Repositories live in `outputs/vendor/native_baselines/`; environment specifications are in `envs/`. Published programs require CUDA. Wanda and FLAP retain their published 7B LLaMA settings; select matching `published_wanda`/`wanda_published` or `published_flap`/`flap_published` presets.

`native.py` validates the source and environment and invokes the external program directly from its Forge run. Commands, package versions, source commits, console output, and any requested model artifacts stay inside that run's artifact directory. These external programs retain their own evaluation pipelines.
