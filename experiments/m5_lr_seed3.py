"""Third seed at each arm's best learning rate.

The lr sweep ran 2 seeds per cell and found the quantized-shadow arm ahead by
2.7% at each arm's own optimum (fp32 21.751 @ lr=0.25, shadow8ef 21.156 @
lr=0.35). Seed spreads were tight -- [21.68, 21.83] and [21.13, 21.18] -- so the
gap is roughly 4x the larger spread, but the project requires 3 seeds minimum
before a comparison is reported. This adds the third.

Output: appends to results/m5_lr_control.csv
"""

from __future__ import annotations

import atexit
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.tinystories import tokenize
from experiments.m5_ladder import train_one
from experiments.repro import allow_sleep, pin, prevent_sleep
from models.quant_mlp import LadderConfig

# each arm at ITS OWN optimum, from the completed sweep
CELLS = [("fp32", 0.25, LadderConfig(name="R0_fp32")),
         ("shadow8ef", 0.35, LadderConfig(name="S5", shadow_bits=8, shadow_mode="ef"))]
SEED = 2


def main() -> int:
    pin()
    if prevent_sleep():
        print("[sleep guard] keep-awake accepted", flush=True)
    atexit.register(allow_sleep)
    data = tokenize()

    out = Path(__file__).resolve().parent.parent / "results" / "m5_lr_control.csv"
    fields = ["arm", "lr", "seed", "val_loss", "val_ppl", "frozen_frac_tail", "cpu_seconds"]
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    done = {(r["arm"], float(r["lr"]), int(r["seed"])) for r in rows}

    import experiments.m3_baseline as B
    import experiments.m5_ladder as M

    for arm, lr, cfg in CELLS:
        if (arm, lr, SEED) in done:
            print(f"  {arm} lr={lr} seed{SEED} already done", flush=True)
            continue
        B.LR_SGD = M.LR_SGD = lr
        r = train_one(cfg, data, SEED)
        rows.append({"arm": arm, "lr": lr, "seed": SEED,
                     "val_loss": r["val_loss"], "val_ppl": r["val_ppl"],
                     "frozen_frac_tail": r["frozen_frac_tail"],
                     "cpu_seconds": r["cpu_seconds"]})
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  {arm:<10} lr={lr} seed{SEED}  ppl {float(r['val_ppl']):7.3f}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
