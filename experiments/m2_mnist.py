"""M2: the ablation ladder on a tiny MLP. Cheap debugging ground for M5.

Runs BOTH designs, as the methodology requires:

  cumulative   R0 -> R1 -> R1+R2 -> ... , the main spine
  solo         each component alone from the FP32 baseline

Cumulative alone confounds: if R3 fails you cannot tell whether gradients break
training, or break training GIVEN that weights are already ternary. Solo alone
misses interactions. You need both.

Fixed budget in STEPS (not epochs, not wall-clock), identical replayed batch
sequence for every run, 3 seeds.

Output: results/m2_mnist.csv, results/m2_mnist_curves.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from data.mnist import frozen_batches, load_mnist
from experiments.repro import pin, threads
from models import LadderConfig, QuantMLP, count_params
from optim import IntSGD
from quant import QuantSpec, registry

STEPS = 1500
BATCH = 128
LR = 0.1
SEEDS = (0, 1, 2)
EVAL_EVERY = 100

# Deeper and more uniform than the obvious (784,256,128,10). With the usual
# BNN convention of leaving the FIRST and LAST layers in full precision, a
# 3-layer net leaves only ONE quantized layer holding 14% of the parameters --
# so nothing breaks and the ladder measures nothing. Measured: R5 full stack
# scored 0.9662 vs 0.9703 fp32 under that design, a difference inside noise.
# Four hidden layers put 39% of parameters inside the ladder.
SIZES = (784, 256, 256, 256, 10)

# --- the specs each rung turns on -------------------------------------------
TERNARY = QuantSpec(kind="ternary", granularity="tensor", calib="absmean")  # BitNet b1.58
ACT8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")     # per-example
ACT4 = QuantSpec(kind="int", bits=4, granularity="row", calib="absmax")
DY8 = QuantSpec(kind="int", bits=8, granularity="row", rounding="stochastic")
DY4 = QuantSpec(kind="int", bits=4, granularity="row", rounding="stochastic")
WG8 = QuantSpec(kind="int", bits=8, granularity="tensor", rounding="stochastic")
WG4 = QuantSpec(kind="int", bits=4, granularity="tensor", rounding="stochastic")


def rungs() -> list[LadderConfig]:
    """The ladder. Includes low-bit variants deliberately: a rung that does not
    break is not informative, so each component is pushed until it does."""
    cumulative = [
        LadderConfig(name="R0_fp32"),
        LadderConfig(name="R1_ternary_w", w_spec=TERNARY),
        LadderConfig(name="R2_+act8", w_spec=TERNARY, a_spec=ACT8),
        LadderConfig(name="R3a_+dgrad8", w_spec=TERNARY, a_spec=ACT8, g_spec=DY8),
        LadderConfig(name="R3b_+wgrad8", w_spec=TERNARY, a_spec=ACT8, g_spec=DY8,
                     wg_spec=WG8),
        LadderConfig(name="R5_+shadow8sr", w_spec=TERNARY, a_spec=ACT8, g_spec=DY8,
                     wg_spec=WG8, shadow_bits=8, shadow_mode="sr"),
        # push: everything at 4 bits
        LadderConfig(name="R5_all4bit", w_spec=TERNARY, a_spec=ACT4, g_spec=DY4,
                     wg_spec=WG4, shadow_bits=4, shadow_mode="sr"),
        # push: quantize EVERY layer including first and last
        LadderConfig(name="R7_alllayers", w_spec=TERNARY, a_spec=ACT8, g_spec=DY8,
                     wg_spec=WG8, shadow_bits=8, shadow_mode="sr",
                     quantize_first_last=True),
    ]
    solo = [
        LadderConfig(name="S2_act8", a_spec=ACT8),
        LadderConfig(name="S2_act4", a_spec=ACT4),
        LadderConfig(name="S3a_dgrad8", g_spec=DY8),
        LadderConfig(name="S3a_dgrad4", g_spec=DY4),
        LadderConfig(name="S3b_wgrad8", wg_spec=WG8),
        LadderConfig(name="S3b_wgrad4", wg_spec=WG4),
        # R5 head-to-head: does the M1-spectrum finding (SR >> EF) reproduce
        # on a real, non-convex model?
        LadderConfig(name="S5_shadow8sr", shadow_bits=8, shadow_mode="sr"),
        LadderConfig(name="S5_shadow8rtn", shadow_bits=8, shadow_mode="rtn"),
        LadderConfig(name="S5_shadow8ef", shadow_bits=8, shadow_mode="ef"),
        LadderConfig(name="S5_shadow4sr", shadow_bits=4, shadow_mode="sr"),
        LadderConfig(name="S5_shadow4rtn", shadow_bits=4, shadow_mode="rtn"),
        LadderConfig(name="S5_shadow4ef", shadow_bits=4, shadow_mode="ef"),
    ]
    return cumulative + solo


@torch.no_grad()
def evaluate(model, x, y, chunk=2000) -> tuple[float, float]:
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(x), chunk):
        xb, yb = x[i:i + chunk], y[i:i + chunk]
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    model.train()
    return loss / len(x), correct / len(x)


def train_one(cfg: LadderConfig, data: dict, idx: torch.Tensor, seed: int) -> dict:
    registry.reset()
    torch.manual_seed(seed)
    model = QuantMLP(cfg, sizes=SIZES, seed=seed)

    shadow_spec = (
        QuantSpec(kind="int", bits=cfg.shadow_bits, granularity="tensor",
                  rounding="stochastic" if cfg.shadow_mode == "sr" else "nearest")
        if cfg.shadow_bits else None
    )
    opt = IntSGD(
        model.parameters(),
        lr=LR,
        shadow_spec=shadow_spec,
        shadow_range="auto" if cfg.shadow_bits else None,
        grad_spec=cfg.wg_spec if cfg.wg_spec.kind != "none" else None,
        error_feedback=(cfg.shadow_mode == "ef" and bool(cfg.shadow_bits)),
        seed=seed,
    )

    xtr, ytr = data["x_train"], data["y_train"]
    xte, yte = data["x_test"], data["y_test"]

    curve, frozen_tail, diverged = [], [], False
    t0 = time.time()
    for t in range(STEPS):
        b = idx[t]
        out = model(xtr[b])
        loss = F.cross_entropy(out, ytr[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if t == 0:
            # THE no-op guard: every enabled rung must have changed something,
            # model-side sites AND optimizer-side (shadow / weight-gradient)
            registry.assert_no_noops()
            opt.assert_quantizers_active()
        if not torch.isfinite(loss):
            diverged = True
            break
        if t >= STEPS - 200:
            frozen_tail.append(opt.last.get("frozen_frac", 0.0))
        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            l, a = evaluate(model, xte, yte)
            curve.append((t, l, a))

    test_loss, test_acc = (float("nan"), 0.0) if diverged else evaluate(model, xte, yte)
    train_loss, train_acc = (float("nan"), 0.0) if diverged else evaluate(
        model, xtr[:10000], ytr[:10000])

    sites = registry.dump_stats()
    return {
        "config": cfg.name,
        "spec": cfg.describe(),
        "seed": seed,
        "steps": STEPS,
        "params": count_params(model),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "diverged": diverged,
        "frozen_frac_tail": (sum(frozen_tail) / len(frozen_tail)) if frozen_tail else 0.0,
        "n_quant_sites": len(sites),
        "max_zero_frac": max([s["zero_frac_mean"] for s in sites], default=0.0),
        "max_sat_frac": max([s["sat_frac_mean"] for s in sites], default=0.0),
        "threads": threads(),
        "seconds": round(time.time() - t0, 1),
        "_curve": curve,
        "_sites": sites,
    }


def main() -> int:
    n = pin()
    data = load_mnist()
    idx = frozen_batches(len(data["x_train"]), BATCH, STEPS, seed=0)
    print(f"[repro] threads={n} | frozen batches {tuple(idx.shape)} | "
          f"budget {STEPS} steps x {BATCH} = {STEPS*BATCH:,} examples")
    print(f"[model] {SIZES}, ReLU, plain SGD lr={LR}, no momentum\n")

    rows, curves = [], []
    for cfg in rungs():
        for seed in SEEDS:
            r = train_one(cfg, data, idx, seed)
            rows.append(r)
            for t, l, a in r["_curve"]:
                curves.append({"config": cfg.name, "seed": seed, "step": t,
                               "test_loss": l, "test_acc": a})
        accs = [x["test_acc"] for x in rows[-len(SEEDS):]]
        print(f"  {cfg.name:<16} acc {sum(accs)/len(accs):.4f} "
              f"[{min(accs):.4f}-{max(accs):.4f}]  "
              f"{rows[-1]['seconds']:.0f}s/run  sites={rows[-1]['n_quant_sites']}")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    fields = [k for k in rows[0] if not k.startswith("_")]
    with (out_dir / "m2_mnist.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with (out_dir / "m2_mnist_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)
    print(f"\nwrote results/m2_mnist.csv ({len(rows)} rows)")

    def agg(name, key):
        v = [r[key] for r in rows if r["config"] == name]
        return sum(v) / len(v), min(v), max(v)

    base, _, _ = agg("R0_fp32", "test_acc")
    print()
    print("=" * 74)
    print(f"{'config':<17} {'test acc':>18} {'vs fp32':>9} {'test loss':>10} {'froz':>6}")
    print("=" * 74)
    for cfg in rungs():
        m, lo, hi = agg(cfg.name, "test_acc")
        l, _, _ = agg(cfg.name, "test_loss")
        fr, _, _ = agg(cfg.name, "frozen_frac_tail")
        print(f"{cfg.name:<17} {m:>7.4f} [{lo:.4f}-{hi:.4f}] {m-base:>+9.4f} "
              f"{l:>10.4f} {fr:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
