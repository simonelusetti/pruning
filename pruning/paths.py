from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
SCRATCH = OUTPUTS / "cache" / "pruning_scratch"
BASELINE_CACHE = OUTPUTS / "cache" / "native_baselines"
MODEL_CACHE = OUTPUTS / "cache" / "huggingface" / "hub"
BASELINE_VENDOR = OUTPUTS / "vendor" / "native_baselines"
MANUAL = OUTPUTS / "manual"
PLOTS = OUTPUTS / "plots"
