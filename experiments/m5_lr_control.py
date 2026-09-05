"""Control for the M5 anomaly: does quantizing the latent weight really BEAT fp32?

The ladder measured S5_shadow8ef_solo at ppl 21.48 against an fp32 baseline of
22.29 -- a 3.7% IMPROVEMENT from quantizing the shadow weight, which should not
be possible. Suspect a confound before believing it.

The confound to rule out: with an 8-bit latent grid, 99.2% of weights do not
move on a given step (measured frozen_frac 0.992). Sparse updates act like a
smaller effective learning rate. The ladder runs at lr=0.5 with momentum 0.9,
which was never tuned. If that lr is simply too high for the fp32 arm, then
"quantization helps" is really "quantization accidentally reduced the step
size", which is an artifact, not a finding.

Decisive test: sweep the fp32 learning rate. If some fp32 lr reaches the
quantized arm's loss, the improvement is an lr artifact and must be reported as
one. If no fp32 lr gets there, the effect survives and is worth a closer look.

Output: results/m5_lr_control.csv
"""

from __future__ import annotations

import atexit
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tinystories import tokenize
from experiments.m5_ladder import STEPS, train_one
from experiments.repro import allow_sleep, pin, prevent_sleep
from models.quant_mlp import LadderConfig

LRS = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0)
SEEDS = (0, 1)


def main() -> int:
    pin()
    if prevent_sleep():
        print("[sleep guard] keep-awake accepted")
    atexit.register(allow_sleep)
    data = tokenize()

    out = Path(__file__).resolve().parent.parent / "results" / "m5_lr_control.csv"
    fields = ["arm", "lr", "seed", "val_loss", "val_ppl", "frozen_frac_tail", "cpu_seconds"]
    rows = []
    if out.exists():
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
    done = {(r["arm"], float(r["lr"]), int(r["seed"])) for r in rows}

    def record(arm, lr, seed, r):
        rows.append({"arm": arm, "lr": lr, "seed": seed,
                     "val_loss": r["val_loss"], "val_ppl": r["val_ppl"],
                     "frozen_frac_tail": r["frozen_frac_tail"],
                     "cpu_seconds": r["cpu_seconds"]})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  {arm:<10} lr={lr:<5} seed{seed}  ppl {float(r['val_ppl']):7.3f}  "
              f"froz {float(r['frozen_frac_tail']):.3f}", flush=True)

    # patch the module-level lr the ladder uses
    import experiments.m3_baseline as B
    import experiments.m5_ladder as M

    print("fp32 arm across learning rates:")
    for lr in LRS:
        for seed in SEEDS:
            if ("fp32", lr, seed) in done:
                continue
            B.LR_SGD = M.LR_SGD = lr
            record("fp32", lr, seed, train_one(LadderConfig(name="R0_fp32"), data, seed))

    print("\nshadow-8bit-EF arm across the same learning rates:")
    for lr in LRS:
        for seed in SEEDS:
            if ("shadow8ef", lr, seed) in done:
                continue
            B.LR_SGD = M.LR_SGD = lr
            record("shadow8ef", lr, seed,
                   train_one(LadderConfig(name="S5", shadow_bits=8, shadow_mode="ef"),
                             data, seed))

    print(f"\nwrote {out}")
    best = {}
    for r in rows:
        k = r["arm"]
        v = float(r["val_ppl"])
        if k not in best or v < best[k][0]:
            best[k] = (v, float(r["lr"]))
    print("\nBEST ppl per arm over the lr sweep:")
    for k, (v, lr) in best.items():
        print(f"  {k:<10} {v:7.3f}  at lr={lr}")
    if "fp32" in best and "shadow8ef" in best:
        if best["fp32"][0] <= best["shadow8ef"][0]:
            print("\n=> CONFOUND CONFIRMED: a tuned fp32 lr matches or beats the")
            print("   quantized arm. The ladder's 'improvement' is an lr artifact.")
        else:
            print("\n=> effect SURVIVES lr tuning: quantized shadow still ahead at")
            print("   every arm's own best lr. Needs a different explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
