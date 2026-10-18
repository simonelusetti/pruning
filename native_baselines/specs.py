"""Pinned source and official quick-start settings for native benchmarks."""
from __future__ import annotations

SPECS = {
    "sparsegpt": {
        "url": "https://github.com/IST-DASLab/sparsegpt.git",
        "commit": "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
        "model": "facebook/opt-125m",
        "dataset": "c4",
        "nsamples": 128,
        "sparsity": 0.5,
        "script": "opt.py",
        "paper": "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot",
        "versions": {"torch": "1.10.1", "transformers": "4.21.2", "datasets": "1.17.0"},
    },
    "wanda": {
        "url": "https://github.com/locuslab/wanda.git",
        "commit": "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6",
        "model": "decapoda-research/llama-7b-hf",
        "dataset": "c4",
        "nsamples": 128,
        "sparsity": 0.5,
        "script": "main.py",
        "paper": "A Simple and Effective Pruning Approach for Large Language Models",
        "versions": {"torch": "1.10.1", "transformers": "4.28.0", "datasets": "2.11.0",
                     "accelerate": "0.18.0"},
    },
    "flap": {
        "url": "https://github.com/CASIA-LMC-Lab/FLAP.git",
        "commit": "3bb57db3449dd2fa04a5c2192de80e87e33be2b1",
        "model": "decapoda-research/llama-7b-hf",
        "dataset": "wikitext2",
        "nsamples": 1024,
        "sparsity": 0.2,
        "script": "main.py",
        "paper": "Fluctuation-Based Adaptive Structured Pruning for Large Language Models",
        "versions": {"torch": "1.10.1", "transformers": "4.28.0", "datasets": "2.11.0",
                     "accelerate": "0.18.0"},
    },
}
