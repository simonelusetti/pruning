# NNpruning

The framework represents pruning operations as moves, screens structural candidates on training data, selects them on validation data, and keeps parameter counting and stopping anchored to the original model.

## Move groups

| Group | MLP moves | Purpose |
|---|---|---|
| `classic` | `unit_sat`, `unit_mean`, `unit_step`, `unit_merge` | Original supervisor framework; this is the default. |
| `outdated` | `unit_ablate` | Retained only for old comparisons. |
| `extension` | quantized Fisher compensation variants | Other current research moves, selectable individually. |
| `fisher` | `unit_fisher` | One compensated deletion per neuron, ranked in the downstream Fisher metric. |

The modular-addition transformer adds `head_fisher` to the `fisher` group and retains QK/OV rank cuts unchanged in both comparisons. The pre-LayerNorm GPT adapter uses the same older group names; its later stream, low-rank, token/head, and K/V-sharing experiments are `outdated`.

`PruneConfig(struct_kinds=None)` selects `classic`. Groups and individual moves can be combined, for example `struct_kinds=("classic", "fisher")` or `struct_kinds=("unit_fisher", "unit_mean")`.

## Maintained suite

- `moons`: small sigmoid MLP for fast correctness and move-selection checks.
- `modadd`: one-layer modular-addition transformer for a larger run that remains practical on CPU.
- `native_baselines/`: pinned launchers for SparseGPT, Wanda, and FLAP in their own model and dataset settings. These require their external checkpoints and normally a CUDA machine.

Cached local starting states live in `checkpoints/`. Each command writes one JSON result containing the metric history, accepted selections in order, and move counts. Use `--output` to choose its path.

```bash
cd NNpruning
python run_experiments.py moons --moves classic fisher --max-loss-error 0.05
python run_experiments.py modadd --moves classic --device cuda --max-loss-error 0.05
```

The adapter and state can also be moved directly:

```python
adapter = adapter.to("cuda")
state = state.to("cuda")
```
