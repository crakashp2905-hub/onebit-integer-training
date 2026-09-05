"""M3: the FP32 reference every rung is compared against.

Records loss vs TOKENS (not epochs, not wall-clock), throughput, and peak
memory. This run also sets the compute budget for the whole ladder, so its
tokens/sec number is a deliverable in its own right.

Two reference arms, deliberately:

  adamw    the standard nanoGPT optimizer. The honest answer to "how good is a
           3M-parameter model on this data".
  intsgd   the optimizer the LADDER actually uses (IntSGD with momentum, FP32
           shadow). Every quantized rung must be compared against THIS, or the
           measured damage is confounded with "SGD is worse than Adam".

Reporting only the first would overstate quantization damage; reporting only the
second would overstate model quality. Both are cheap relative to the ladder.

Usage:
  python experiments/m3_baseline.py --quick     200 steps, 1 seed, sanity check
  python experiments/m3_baseline.py             full budget, 3 seeds
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from data.tinystories import frozen_batches, get_batch, tokenize
from experiments.repro import pin, threads
from models import GPTConfig, QuantGPT
from optim import IntSGD

BATCH = 16
CTX = 128
STEPS = 1000
LR_ADAMW = 1e-3
LR_SGD = 0.5
MOMENTUM = 0.9
WARMUP = 100
EVAL_EVERY = 50
EVAL_BATCHES = 20
SEEDS = (0, 1, 2)


def model_config(ladder=None) -> GPTConfig:
    from models.quant_mlp import LadderConfig

    return GPTConfig(
        vocab_size=2048, ctx=CTX, n_layer=6, n_head=6, d_model=192,
        ladder=ladder or LadderConfig(),
    )


def lr_at(step: int, base: float, total: int) -> float:
    """Linear warmup then cosine decay to 10%."""
    if step < WARMUP:
        return base * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, total - WARMUP)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, val_tokens, ctx, n_batches=EVAL_BATCHES, seed=1234):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(val_tokens) - ctx - 1, (n_batches, BATCH), generator=g)
    tot = 0.0
    for i in range(n_batches):
        x, y = get_batch(val_tokens, offs[i], ctx)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / n_batches


def peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def run(opt_name: str, seed: int, steps: int, data: dict) -> dict:
    torch.manual_seed(seed)
    cfg = model_config()
    model = QuantGPT(cfg, seed=seed)
    pb = model.param_breakdown()

    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=LR_ADAMW, betas=(0.9, 0.95),
                                weight_decay=0.1)
        base_lr = LR_ADAMW
    else:
        opt = IntSGD(model.parameters(), lr=LR_SGD, momentum=MOMENTUM, seed=seed)
        base_lr = LR_SGD

    train, val = data["train"], data["val"]
    offsets = frozen_batches(len(train), BATCH, CTX, steps, seed=0)

    curve = []
    t0 = time.time()
    tokens = 0
    for t in range(steps):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(t, base_lr, steps)
        x, y = get_batch(train, offsets[t], CTX)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.numel()
        if t % EVAL_EVERY == 0 or t == steps - 1:
            vl = evaluate(model, val, CTX)
            curve.append((t, tokens, loss.item(), vl))

    elapsed = time.time() - t0
    val_loss = evaluate(model, val, CTX, n_batches=50)
    return {
        "optimizer": opt_name,
        "seed": seed,
        "steps": steps,
        "tokens": tokens,
        "params_total": pb["total"],
        "params_body": pb["body"],
        "body_frac": round(pb["body_frac"], 4),
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "tokens_per_sec": round(tokens / elapsed, 1),
        "seconds": round(elapsed, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "threads": threads(),
        "_curve": curve,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    steps = 200 if args.quick else STEPS
    seeds = (0,) if args.quick else SEEDS

    pin()
    data = tokenize()
    cfg = model_config()
    pb = QuantGPT(cfg).param_breakdown()
    print(f"[data]  vocab {data['vocab_size']} | train {len(data['train']):,} tokens")
    print(f"[model] d={cfg.d_model} L={cfg.n_layer} H={cfg.n_head} ctx={cfg.ctx}")
    print(f"        {pb['total']:,} params | body {pb['body']:,} ({pb['body_frac']:.1%}) "
          f"| embedding {pb['embedding']:,}")
    print(f"[budget] {steps} steps x {BATCH} x {CTX} = {steps*BATCH*CTX:,} tokens")
    print(f"[random baseline] ln(vocab) = {math.log(data['vocab_size']):.4f} nats\n")

    rows, curves = [], []
    for opt_name in ("adamw", "intsgd"):
        for seed in seeds:
            r = run(opt_name, seed, steps, data)
            rows.append(r)
            for st, tk, tl, vl in r["_curve"]:
                curves.append({"optimizer": opt_name, "seed": seed, "step": st,
                               "tokens": tk, "train_loss": tl, "val_loss": vl})
            print(f"  {opt_name:<7} seed{seed}  val_loss {r['val_loss']:.4f}  "
                  f"ppl {r['val_ppl']:7.2f}  {r['tokens_per_sec']:>7.0f} tok/s  "
                  f"{r['seconds']/60:.1f} min  rss {r['peak_rss_mb']:.0f} MB")

    out = Path(__file__).resolve().parent.parent / "results"
    fields = [k for k in rows[0] if not k.startswith("_")]
    suffix = "_quick" if args.quick else ""
    with (out / f"m3_baseline{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with (out / f"m3_baseline_curves{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)

    print()
    for opt_name in ("adamw", "intsgd"):
        v = [r["val_loss"] for r in rows if r["optimizer"] == opt_name]
        tps = [r["tokens_per_sec"] for r in rows if r["optimizer"] == opt_name]
        print(f"  {opt_name:<7} val_loss {sum(v)/len(v):.4f} "
              f"[{min(v):.4f}-{max(v):.4f}]  ppl {math.exp(sum(v)/len(v)):.2f}  "
              f"{sum(tps)/len(tps):.0f} tok/s")

    tps = sum(r["tokens_per_sec"] for r in rows) / len(rows)
    budget = steps * BATCH * CTX
    print(f"\n[LADDER COST PROJECTION] at {tps:.0f} tok/s FP32, one run of "
          f"{budget:,} tokens = {budget/tps/60:.0f} min.")
    print(f"  fake quant is ~2-3x slower, so a quantized run is ~"
          f"{budget/tps/60*2:.0f}-{budget/tps/60*3:.0f} min.")
    print(f"  a 10-rung ladder at 3 seeds = 30 runs = "
          f"{30*budget/tps/60*2.5/60:.0f} hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
