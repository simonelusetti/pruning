from __future__ import annotations
import random
from functools import lru_cache
from pathlib import Path
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from pruning.paths import BASELINE_CACHE
DATASET_CACHE = BASELINE_CACHE / "datasets"
EVALUATION_CACHE = BASELINE_CACHE / "evaluation"


@lru_cache(maxsize=1)
def c4_training_data():
    try:
        return load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            cache_dir=str(DATASET_CACHE),
        )
    except ValueError:
        shards = sorted((DATASET_CACHE / "allenai___c4").glob("**/c4-train-*.arrow"))
        if not shards:
            raise
        return concatenate_datasets([Dataset.from_file(str(path)) for path in shards])


def calibration_tokens(tokenizer, dataset: str, nsamples: int, seqlen: int,
                       seed: int) -> list[torch.Tensor]:
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          cache_dir=str(DATASET_CACHE))
        encoded = tokenizer(" ".join(ds["text"]), return_tensors="pt").input_ids
        rng = random.Random(seed)
        return [encoded[:, (i := rng.randint(0, encoded.shape[1] - seqlen - 1)):i + seqlen]
                for _ in range(nsamples)]

    ds = c4_training_data()
    rng = random.Random(seed)
    samples = []
    byte_limit = (seqlen - tokenizer.num_special_tokens_to_add(pair=False)
                  - int(getattr(tokenizer, "add_prefix_space", False))
                  if hasattr(tokenizer, "byte_encoder") else None)
    while len(samples) < nsamples:
        text = ds[rng.randrange(len(ds))]["text"]
        # Byte-level BPE emits at most one ordinary token per UTF-8 byte.
        if byte_limit is not None and len(text.encode("utf-8")) <= byte_limit:
            continue
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[:, start:start + seqlen])
    return samples


def wikitext_test_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    return ids if max_tokens <= 0 else ids[:, :max_tokens]


def official_test_tokens(tokenizer, dataset: str, seqlen: int) -> torch.Tensor:
    if dataset == "wikitext2":
        return wikitext_test_tokens(tokenizer, 0)
    if dataset == "ptb":
        ds = Dataset.from_file(str(EVALUATION_CACHE / "ptb_test.arrow"))
        return tokenizer(" ".join(ds["text"]), return_tensors="pt").input_ids
    ds = Dataset.from_file(str(EVALUATION_CACHE / "c4_validation.arrow"))
    ids = tokenizer(" ".join(ds[:1100]["text"]), return_tensors="pt").input_ids
    return ids[:, :256 * seqlen]


def wikitext_validation_sequences(tokenizer, seqlen: int, offset: int,
                                  count: int) -> list[torch.Tensor]:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation",
                      cache_dir=str(DATASET_CACHE))
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    end = (offset + count) * seqlen
    if ids.numel() < end:
        raise ValueError(f"validation split has {ids.numel()} tokens, need {end}")
    return [ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0)
            for i in range(offset, offset + count)]


def selection_sequences(tokenizer, dataset: str, seqlen: int, offset: int,
                        count: int, seed: int) -> list[torch.Tensor]:
    if dataset == "wikitext2":
        return wikitext_validation_sequences(tokenizer, seqlen, offset, count)
    samples = calibration_tokens(tokenizer, "c4", offset + count, seqlen, seed)
    return samples[offset:]
