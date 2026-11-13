from __future__ import annotations

import torch


def disk_kth(paths, rank, *, device="cpu", chunk_size=8_000_000):
    prefix = 0
    for shift in range(56, -1, -8):
        histogram = torch.zeros(256, dtype=torch.int64, device=device)
        for path in paths:
            values = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            for chunk in values.flatten().split(chunk_size):
                chunk = chunk.to(device=device, dtype=torch.float64)
                bits = chunk.view(torch.int64)
                eligible = chunk.isfinite()
                if shift < 56:
                    eligible &= (bits >> (shift+8)) == (prefix >> (shift+8))
                bins = (bits[eligible] >> shift) & 255
                histogram += torch.bincount(bins, minlength=256)
        cumulative = histogram.cumsum(0)
        if not 1 <= rank <= int(cumulative[-1]):
            raise ValueError("Requested score rank exceeds the finite nonnegative scores")
        bucket = int(torch.searchsorted(cumulative, rank))
        rank -= int(cumulative[bucket-1]) if bucket else 0
        prefix |= bucket << shift
    return float(torch.tensor(prefix, dtype=torch.int64).view(torch.float64)), rank


def disk_masks(paths, rank, output, *, device="cpu", chunk_size=8_000_000):
    threshold, ties = disk_kth(paths, rank, device=device, chunk_size=chunk_size)
    masks, total, selected = [], 0., 0
    for index, path in enumerate(paths):
        values = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        mask = torch.empty_like(values, dtype=torch.bool, device="cpu")
        offset = 0
        for chunk in values.flatten().split(chunk_size):
            chunk = chunk.to(device)
            chosen = chunk < threshold
            equal = chunk == threshold
            take = min(ties, int(equal.sum()))
            if take:
                chosen |= equal & (equal.cumsum(0) <= take)
            ties -= take
            mask.flatten()[offset:offset+len(chunk)] = chosen.cpu()
            total += float(chunk[chosen].sum(dtype=torch.float64))
            selected += int(chosen.sum())
            offset += len(chunk)
        destination = output / f"mask-{index}.pt"
        torch.save(mask, destination)
        masks.append(destination)
    if selected != rank or ties:
        raise RuntimeError("Disk selection did not produce the requested deletion count")
    return masks, total
