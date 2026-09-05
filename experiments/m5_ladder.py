"""M5: the ablation ladder on the 3.1M-parameter decoder.

Budget: 1000 steps x 16 x 128 = 2,048,000 tokens per run, identical replayed
batch sequence, 3 seeds, IntSGD + momentum 0.9 (matching the M3 intsgd arm, so
quantization damage is never confounded with "SGD is worse than Adam").

RESULTS ARE APPENDED AFTER EVERY RUN. This job takes ~21 hours; a crash at hour
19 must not lose 19 hours of work. Re-running skips configs already present in
the CSV, so it resumes.

A PREDICTION TO TEST, recorded before the run (M2 used momentum=0, this uses
0.9): momentum is itself an accumulator. The buffer sums small gradients until
lr*buf crosses delta/2, which is error feedback in the gradient domain. So RTN
latent weights may NOT collapse here the way they did at M2. If they survive,
momentum is a partial fix for swamping and that is a finding in its own right;
if they still collapse, the freeze is more robust than momentum can repair.

Canaries recorded per run: frozen_frac (the M1 freeze canary), gradient
zero_frac (silent underflow), saturation fraction.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.tinystories import frozen_batches, get_batch, tokenize
from experiments.m3_baseline import BATCH, CTX, EVAL_BATCHES, LR_SGD, MOMENTUM, WARMUP, lr_at
from experiments.repro import allow_sleep, cpu_clock, pin, prevent_sleep, threads
from models import GPTConfig, QuantGPT
from models.quant_mlp import LadderConfig
from optim import IntSGD
from quant import QuantSpec, registry

STEPS = 1000
SEEDS = (0, 1, 2)
EVAL_EVERY = 100

# ---------------------------------------------------------------- the specs
TERN = QuantSpec(kind="ternary", granularity="tensor", calib="absmean")
TERN_P2 = QuantSpec(kind="ternary", granularity="tensor", calib="absmean",
                    scale_mode="pow2", pow2_mode="ceil")
ACT8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")
ACT8_P2 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax",
                    scale_mode="pow2", pow2_mode="ceil")
ACT4 = QuantSpec(kind="int", bits=4, granularity="row", calib="absmax")
DY8 = QuantSpec(kind="int", bits=8, granularity="row", rounding="stochastic")
WG8 = QuantSpec(kind="int", bits=8, granularity="tensor", rounding="stochastic")


def ladder() -> list[LadderConfig]:
    """Cumulative spine, plus the multiplier-free probe, the R5 head-to-head,
    and one solo control that isolates R5 from everything else."""
    full = dict(w_spec=TERN, a_spec=ACT8, g_spec=DY8, wg_spec=WG8)
    return [
        LadderConfig(name="R0_fp32"),
        LadderConfig(name="R1_ternary", w_spec=TERN),
        LadderConfig(name="R2_act8", w_spec=TERN, a_spec=ACT8),
        # the multiplier-free probe: same bits, power-of-two scales only
        LadderConfig(name="R2p5_pow2scales", w_spec=TERN_P2, a_spec=ACT8_P2),
        LadderConfig(name="R2p9_act4", w_spec=TERN, a_spec=ACT4),
        LadderConfig(name="R3a_dgrad8", w_spec=TERN, a_spec=ACT8, g_spec=DY8),
        LadderConfig(name="R3b_wgrad8", **full),
        # R5 head-to-head on top of the full forward+backward integer stack
        LadderConfig(name="R5_shadow8ef", **full, shadow_bits=8, shadow_mode="ef"),
        LadderConfig(name="R5_shadow8sr", **full, shadow_bits=8, shadow_mode="sr"),
        LadderConfig(name="R5_shadow8rtn", **full, shadow_bits=8, shadow_mode="rtn"),
        LadderConfig(name="R5_shadow4ef", **full, shadow_bits=4, shadow_mode="ef"),
        # solo control: shadow weights only, everything else FP32
        LadderConfig(name="S5_shadow8ef_solo", shadow_bits=8, shadow_mode="ef"),
    ]


@torch.no_grad()
def evaluate(model, val, n_batches=EVAL_BATCHES, seed=1234):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    offs = torch.randint(0, len(val) - CTX - 1, (n_batches, BATCH), generator=g)
    tot = 0.0
    for i in range(n_batches):
        x, y = get_batch(val, offs[i], CTX)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / n_batches


def train_one(cfg: LadderConfig, data: dict, seed: int) -> dict:
    registry.reset()
    torch.manual_seed(seed)
    gcfg = GPTConfig(vocab_size=2048, ctx=CTX, n_layer=6, n_head=6, d_model=192,
                     ladder=cfg)
    model = QuantGPT(gcfg, seed=seed)

    shadow_spec = (
        QuantSpec(kind="int", bits=cfg.shadow_bits, granularity="tensor",
                  rounding="stochastic" if cfg.shadow_mode == "sr" else "nearest")
        if cfg.shadow_bits else None
    )
    opt = IntSGD(
        model.parameters(), lr=LR_SGD, momentum=MOMENTUM,
        shadow_spec=shadow_spec,
        shadow_range="auto" if cfg.shadow_bits else None,
        grad_spec=cfg.wg_spec if cfg.wg_spec.kind != "none" else None,
        error_feedback=(cfg.shadow_mode == "ef" and bool(cfg.shadow_bits)),
        seed=seed,
    )

    train, val = data["train"], data["val"]
    offsets = frozen_batches(len(train), BATCH, CTX, STEPS, seed=0)

    curve, frozen_tail, gz_tail = [], [], []
    diverged = False
    tokens = 0
    t0 = time.time()
    c0 = cpu_clock()
    for t in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(t, LR_SGD, STEPS)
        x, y = get_batch(train, offsets[t], CTX)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens += x.numel()

        if t == 0:
            # model-side sites AND optimizer-side (shadow / weight-gradient)
            registry.assert_no_noops()
            opt.assert_quantizers_active()
        if not torch.isfinite(loss):
            diverged = True
            break
        if t >= STEPS - 200:
            frozen_tail.append(opt.last.get("frozen_frac", 0.0))
            gz_tail.append(opt.last.get("grad_zero_frac", 0.0))
        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            curve.append((t, tokens, loss.item(), evaluate(model, val)))

    elapsed = time.time() - t0
    cpu_elapsed = cpu_clock() - c0
    val_loss = float("nan") if diverged else evaluate(model, val, n_batches=50)
    sites = registry.dump_stats()
    return {
        "config": cfg.name,
        "spec": cfg.describe(),
        "seed": seed,
        "steps": STEPS,
        "tokens": tokens,
        "val_loss": val_loss,
        "val_ppl": float("nan") if diverged else math.exp(val_loss),
        "diverged": diverged,
        "frozen_frac_tail": sum(frozen_tail) / len(frozen_tail) if frozen_tail else 0.0,
        "grad_zero_frac_tail": sum(gz_tail) / len(gz_tail) if gz_tail else 0.0,
        "n_quant_sites": len(sites),
        "max_zero_frac": max([s["zero_frac_mean"] for s in sites], default=0.0),
        "max_sat_frac": max([s["sat_frac_mean"] for s in sites], default=0.0),
        # cpu_seconds is the trustworthy one: wall clock includes machine sleep
        "tokens_per_cpu_sec": round(tokens / cpu_elapsed, 1) if cpu_elapsed > 0 else 0.0,
        "cpu_seconds": round(cpu_elapsed, 1),
        "tokens_per_sec": round(tokens / elapsed, 1),
        "seconds": round(elapsed, 1),
        "wall_over_cpu": round(elapsed / cpu_elapsed, 2) if cpu_elapsed > 0 else 0.0,
        "threads": threads(),
        "_curve": curve,
    }


FIELDS = ["config", "spec", "seed", "steps", "tokens", "val_loss", "val_ppl",
          "diverged", "frozen_frac_tail", "grad_zero_frac_tail", "n_quant_sites",
          "max_zero_frac", "max_sat_frac", "tokens_per_cpu_sec", "cpu_seconds",
          "tokens_per_sec", "seconds", "wall_over_cpu", "threads"]


def done_already(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["config"], int(r["seed"])) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated config names")
    args = ap.parse_args()

    pin()
    awake = prevent_sleep()
    atexit.register(allow_sleep)
    print(f"[sleep guard] keep-awake request {'ACCEPTED' if awake else 'NOT AVAILABLE'} "
          f"-- scoped to this process, power plan untouched")
    data = tokenize()
    # ONEBIT_RESULTS lets a read-only checkout (e.g. a Kaggle dataset mount)
    # write results somewhere writable without editing the experiment.
    out = Path(os.environ.get("ONEBIT_RESULTS",
                              Path(__file__).resolve().parent.parent / "results"))
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "m5_ladder.csv"
    curve_path = out / "m5_ladder_curves.csv"
    done = done_already(csv_path)

    configs = ladder()
    if args.only:
        want = set(args.only.split(","))
        configs = [c for c in configs if c.name in want]

    todo = [(c, s) for c in configs for s in SEEDS if (c.name, s) not in done]
    print(f"[m5] {len(configs)} configs x {len(SEEDS)} seeds; "
          f"{len(done)} already done, {len(todo)} to run")
    print(f"[m5] budget {STEPS*BATCH*CTX:,} tokens/run, threads={threads()}\n")

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        with curve_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["config", "seed", "step", "tokens",
                                          "train_loss", "val_loss"]).writeheader()

    t_start = time.time()
    for i, (cfg, seed) in enumerate(todo):
        r = train_one(cfg, data, seed)
        # append immediately -- this job is long enough that a crash must not
        # cost more than one run
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({k: r[k] for k in FIELDS})
        with curve_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config", "seed", "step", "tokens",
                                              "train_loss", "val_loss"])
            for st, tk, tl, vl in r["_curve"]:
                w.writerow({"config": cfg.name, "seed": seed, "step": st,
                            "tokens": tk, "train_loss": tl, "val_loss": vl})
        el = time.time() - t_start
        eta = el / (i + 1) * (len(todo) - i - 1) / 3600
        print(f"  [{i+1}/{len(todo)}] {cfg.name:<20} seed{seed} "
              f"ppl {r['val_ppl']:8.2f} froz {r['frozen_frac_tail']:.3f} "
              f"{r['cpu_seconds']/60:5.1f} cpu-min ({r['seconds']/60:.0f} wall) "
              f"ETA {eta:.1f} h*", flush=True)
    print(f"\ndone in {(time.time()-t_start)/3600:.2f} h -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
