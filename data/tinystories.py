"""TinyStories: download once, train a small BPE, tokenize once, freeze.

Two deliberate choices, both driven by earlier measurements:

1. SMALL VOCAB. At M2 the first MLP design put 86% of its parameters outside the
   ladder and consequently measured nothing. The same trap is worse for a
   transformer: with a 50k GPT-2 vocab and d_model=192, the embedding table is
   9.6M parameters and the transformer body is 2.7M -- the "model" would be
   mostly a lookup table, and every rung would really be measuring the
   embedding. A 2048-token BPE puts ~87% of parameters in the body.

2. VALIDATION SPLIT ONLY. The full TinyStories train file is ~2 GB. The
   validation file is ~22 MB, which is plenty of tokens for a 1-3M parameter
   model on a CPU budget, and this machine has limited disk.

Everything is frozen to disk: the tokenizer, the token array, and the batch
index matrix. Every run in the ladder replays byte-identical batches.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
# ONEBIT_CACHE lets the code run from a read-only mount (a Kaggle
# dataset is mounted read-only, so writing the cache next to the
# module raises OSError: Read-only file system).
CACHE = Path(os.environ.get("ONEBIT_CACHE", ROOT / "cache"))
URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
RAW = CACHE / "tinystories_valid.txt"
VOCAB_SIZE = 2048


def download() -> Path:
    if RAW.exists() and RAW.stat().st_size > 1_000_000:
        return RAW
    import requests

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    RAW.write_bytes(r.content)
    print(f"  wrote {RAW} ({RAW.stat().st_size/1e6:.1f} MB)")
    return RAW


def train_tokenizer(vocab_size: int = VOCAB_SIZE):
    """Byte-level BPE trained on this corpus. Cached to disk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    p = CACHE / f"bpe_{vocab_size}.json"
    if p.exists():
        return Tokenizer.from_file(str(p))
    path = download()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(path)], trainer)
    tok.save(str(p))
    return tok


def tokenize(vocab_size: int = VOCAB_SIZE) -> dict:
    """Tokenize the corpus ONCE and cache as uint16."""
    p = CACHE / f"tokens_{vocab_size}.npz"
    if p.exists():
        z = np.load(p)
        return {"train": z["train"], "val": z["val"], "vocab_size": int(z["vocab_size"])}

    tok = train_tokenizer(vocab_size)
    text = download().read_text(encoding="utf-8", errors="ignore")
    ids = np.array(tok.encode(text).ids, dtype=np.uint16)
    n_val = min(200_000, len(ids) // 10)
    train, val = ids[:-n_val], ids[-n_val:]
    np.savez_compressed(p, train=train, val=val, vocab_size=vocab_size)
    return {"train": train, "val": val, "vocab_size": vocab_size}


def frozen_batches(n_tokens: int, batch: int, ctx: int, steps: int, seed: int = 0):
    """[steps, batch] matrix of start offsets, built once and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"ts_batches_b{batch}_c{ctx}_s{steps}_seed{seed}.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n_tokens - ctx - 1, (steps, batch), generator=g)
    torch.save(idx, p)
    return idx


def get_batch(tokens: np.ndarray, offsets: torch.Tensor, ctx: int):
    """(x, y) for one step. y is x shifted by one."""
    o = offsets.numpy()
    x = np.stack([tokens[i : i + ctx] for i in o]).astype(np.int64)
    y = np.stack([tokens[i + 1 : i + 1 + ctx] for i in o]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)
