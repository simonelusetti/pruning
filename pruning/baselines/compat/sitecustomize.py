"""Loaded only by pinned native baselines. The current allenai/c4 metadata lists splits that
datasets==2.11 then requires in every single-file load; skipping that count check leaves the
files and their contents unchanged."""
import functools

import datasets

_load_dataset = datasets.load_dataset


@functools.wraps(_load_dataset)
def load_dataset(*args, **kwargs):
    kwargs.setdefault("verification_mode", "no_checks")
    return _load_dataset(*args, **kwargs)


datasets.load_dataset = load_dataset
