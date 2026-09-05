"""MNIST with a FROZEN batch sequence.

Methodology requirement: every run in the ladder must see the identical sequence
of batches, so that data variance can never masquerade as a precision effect.
For a language model that means tokenize-once-and-freeze; the MNIST analogue is
to precompute the entire [steps, batch_size] index matrix once, write it to
disk, and replay it verbatim in every run.

The index matrix is saved rather than regenerated so it is inspectable and so a
change to RNG behaviour in a future torch version cannot silently alter it.

Normalization statistics are computed once from the training split and frozen
into the cache, for the same reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
# ONEBIT_CACHE lets the code run from a read-only mount (a Kaggle
# dataset is mounted read-only, so writing the cache next to the
# module raises OSError: Read-only file system).
CACHE = Path(os.environ.get("ONEBIT_CACHE", ROOT / "cache"))


def _download_and_cache() -> dict:
    from torchvision import datasets

    CACHE.mkdir(parents=True, exist_ok=True)
    tr = datasets.MNIST(str(CACHE / "raw"), train=True, download=True)
    te = datasets.MNIST(str(CACHE / "raw"), train=False, download=True)

    xtr = tr.data.to(torch.float32).div_(255.0).reshape(len(tr), -1)
    xte = te.data.to(torch.float32).div_(255.0).reshape(len(te), -1)
    mean = xtr.mean().item()
    std = xtr.std().item()
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    blob = {
        "x_train": xtr,
        "y_train": tr.targets.clone(),
        "x_test": xte,
        "y_test": te.targets.clone(),
        "mean": mean,
        "std": std,
    }
    torch.save(blob, CACHE / "mnist.pt")
    return blob


def load_mnist() -> dict:
    """Load the frozen tensor cache, downloading once if absent (~11 MB)."""
    p = CACHE / "mnist.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    return _download_and_cache()


def frozen_batches(n_examples: int, batch_size: int, steps: int, seed: int = 0) -> torch.Tensor:
    """The [steps, batch_size] index matrix, built once and cached to disk.

    Sampling is with replacement from a dedicated generator, so it never touches
    the global RNG stream and is independent of anything a model does.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"batches_n{n_examples}_b{batch_size}_s{steps}_seed{seed}.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_examples, (steps, batch_size), generator=g)
    torch.save(idx, p)
    return idx


def describe(blob: dict) -> str:
    return (
        f"MNIST: train {tuple(blob['x_train'].shape)} test {tuple(blob['x_test'].shape)} "
        f"| normalized mean={blob['mean']:.4f} std={blob['std']:.4f} "
        f"| train mean {blob['x_train'].mean():.2e} std {blob['x_train'].std():.4f}"
    )
