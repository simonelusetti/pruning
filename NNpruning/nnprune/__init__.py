"""Boundary pruning for neural networks (port of the emergentSR geometry+boundary idea)."""
from .core import (ACTIVE, REMOVED, STEP, Adapter, Move, PruneConfig, State,
                   baseline_path, paired_stats, prune_path, select_move, summarize)
