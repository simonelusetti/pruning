from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
BASELINE_CACHE = OUTPUTS / "cache" / "native_baselines"
BASELINE_VENDOR = OUTPUTS / "vendor" / "native_baselines"
MANUAL = OUTPUTS / "manual"
PLOTS = OUTPUTS / "plots"
