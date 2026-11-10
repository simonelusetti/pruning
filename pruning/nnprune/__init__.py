"""Boundary pruning for neural networks (port of the emergentSR geometry+boundary idea)."""
from .core import (ACTIVE, REMOVED, STEP, Adapter, Move, PruneConfig, State,
                   paired_stats, select_move)
