# Repository rules

- Keep the framework small. Add comments only when they explain a non-obvious invariant needed for maintenance.
- Preserve top-level `sources/` as the reference archive for papers, original code, and project source notes. Consult it before implementing, reproducing, or describing published methods; distinguish published claims from project notes. `sources/original code/` is archival reference, not maintained framework code. Do not remove this archive during code cleanups.
- This is research code: do not add or run automated tests by default. Use them only when the user explicitly requests them or there is genuine uncertainty about how a piece of code will run; keep any such check narrowly focused.
- Do not save experiment tables, logs, reports, or notes unless the user explicitly asks for an artifact.
- Keep tensor work in PyTorch and batch data with `torch.utils.data.DataLoader`; use Python loops only for structurally different candidates or batches.
- New code must run on CPU or CUDA through an explicit device, without device-specific branches in the algorithms.
- Maintained methods are `classic` (S or W+S), `joint` (model-independent Fisher MIQP with inferred safe canonicalization), and `singular` (progressive AXB full-complement updates for OPT).
- The maintained local benchmarks are two-moons for quick checks and modular addition for minute-scale checks. Native SparseGPT, Wanda, and FLAP scripts retain their published model/data settings.
- Do not add CNN or Fashion-MNIST code.
- Launch long-running experiments in a detached `tmux` session and verify the worker remains active after launch.
- Work on `main`. Do not create Git branches or commits; leave changes uncommitted unless the user explicitly reverses this rule.
