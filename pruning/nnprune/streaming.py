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
    live on disk. Float64 bits become order-preserving unsigned keys: positives flip the sign bit, negatives flip all."""
    prefix, sign = 0, -(1 << 63)
    for shift in range(56, -1, -8):
        histogram = torch.zeros(256, dtype=torch.float64, device=device)
        for values, weights in chunks():
            values = values.flatten()
            weights = torch.ones_like(values, dtype=torch.float64) if weights is None else weights.flatten()
            for value, weight in zip(values.split(chunk_size), weights.split(chunk_size)):
                bits = value.to(device=device, dtype=torch.float64).view(torch.int64)
                keys = bits ^ ((bits >> 63) | sign)
                eligible = value.to(device).isfinite()
                if shift < 56:
                    eligible &= ((keys >> (shift + 8)) & ((1 << (56 - shift)) - 1)) == (prefix >> (shift + 8))
                histogram += torch.bincount((keys[eligible] >> shift) & 255,
                                            weights=weight.to(device=device, dtype=torch.float64)[eligible], minlength=256)
        cumulative = histogram.cumsum(0)
        if rank > float(cumulative[-1]):
            raise ValueError("Requested removal exceeds the eligible candidates")
        bucket = int(torch.searchsorted(cumulative, cumulative.new_tensor(rank)))
        rank -= float(cumulative[bucket - 1]) if bucket else 0.
        prefix |= bucket << shift
    key = prefix - (1 << 64) if prefix >= 1 << 63 else prefix
    return float(torch.tensor(key ^ sign if key < 0 else ~key, dtype=torch.int64).view(torch.float64))
