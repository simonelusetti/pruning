from __future__ import annotations

from pathlib import Path
import torch


class Store:
    """Tensors kept in memory, or spilled to a scratch directory when a model's factors exceed memory."""
    def __init__(self, directory: Path | None = None):
        self.directory, self.items = directory, {}

    def __setitem__(self, key, value):
        if self.directory is None:
            self.items[key] = value
        else:
            path = self.directory / f"{len(self.items)}.pt"
            torch.save(value, path)
            self.items[key] = path

    def get(self, key, device):
        value = self.items[key]
        return value if self.directory is None else torch.load(value, map_location=device, weights_only=True)

    def __contains__(self, key):
        return key in self.items


def kth(chunks, rank, *, device="cpu", chunk_size=8_000_000):
    """Smallest threshold whose weighted count of finite values reaches `rank`.

    `chunks()` yields (values, weights or None) pairs and is called once per radix pass, so the scores can
    live on disk. Nonnegative float64 bit patterns sort like their values."""
    prefix = 0
    for shift in range(56, -1, -8):
        histogram = torch.zeros(256, dtype=torch.float64, device=device)
        for values, weights in chunks():
            values = values.flatten()
            weights = torch.ones_like(values, dtype=torch.float64) if weights is None else weights.flatten()
            for value, weight in zip(values.split(chunk_size), weights.split(chunk_size)):
                bits = value.to(device=device, dtype=torch.float64).view(torch.int64)
                eligible = value.to(device).isfinite()
                if shift < 56:
                    eligible &= (bits >> (shift + 8)) == (prefix >> (shift + 8))
                histogram += torch.bincount((bits[eligible] >> shift) & 255,
                                            weights=weight.to(device=device, dtype=torch.float64)[eligible], minlength=256)
        cumulative = histogram.cumsum(0)
        if rank > float(cumulative[-1]):
            raise ValueError("Requested removal exceeds the eligible candidates")
        bucket = int(torch.searchsorted(cumulative, cumulative.new_tensor(rank)))
        rank -= float(cumulative[bucket - 1]) if bucket else 0.
        prefix |= bucket << shift
    return float(torch.tensor(prefix, dtype=torch.int64).view(torch.float64))
