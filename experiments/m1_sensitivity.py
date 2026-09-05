"""How much does a roundoff-level perturbation change an M1 result?

Discovered while benchmarking: torch.linalg.qr is thread-count dependent, so the
SAME problem built under 1 vs 4 torch threads differs by ~1e-6 relative -- pure
float32 roundoff. That gives a free, physically meaningful perturbation with
which to measure how chaotic each rounding mode is.

This matters because quantization is DISCONTINUOUS: a perturbation far below the
grid step can still flip a rounding decision, and over thousands of steps that
can compound. If the amplification is large, then effect sizes smaller than the
amplified noise are NOT resolved by 3-seed spread, and any conclusion resting on
such a margin is unsupported.

Output: results/m1_sensitivity.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments.convex_problem import make_quadratic
from experiments.m1_convex import BIT_GRID, MODES, run
from experiments.repro import pin

D = 1000
STEPS = 3000
KAPPA = 100.0


def main() -> int:
    # Build the SAME problem two ways: the only difference is LAPACK's thread
    # count during QR, i.e. a pure roundoff perturbation.
    torch.set_num_threads(1)
    p_a = make_quadratic(D, KAPPA, seed=0)
    torch.set_num_threads(4)
    p_b = make_quadratic(D, KAPPA, seed=0)

    pin(4)  # from here on, ONLY the problem differs

    rel_in = (p_a.A - p_b.A).abs().max().item() / p_a.A.abs().max().item()
    print(f"input perturbation (relative, in A): {rel_in:.3e}")
    print("(this is float32 roundoff -- ~6 ulp at |A|max)\n")

    rows = []
    print(f"  {'bits':>4} {'mode':>4} {'rel_gap(A)':>13} {'rel_gap(B)':>13} "
          f"{'rel diff':>10} {'amplification':>14}")
    for bits in BIT_GRID:
        for mode in MODES:
            a = run(p_a, bits=bits, mode=mode, steps=STEPS, seed=0,
                    check_freeze_law=False)["rel_gap"]
            b = run(p_b, bits=bits, mode=mode, steps=STEPS, seed=0,
                    check_freeze_law=False)["rel_gap"]
            rd = abs(a - b) / b if b else 0.0
            rows.append({
                "kappa": KAPPA, "bits": bits, "mode": mode,
                "rel_gap_a": a, "rel_gap_b": b,
                "rel_diff": rd, "amplification": rd / rel_in,
                "input_perturbation": rel_in,
            })
            print(f"  {bits:>4} {mode:>4} {a:>13.6e} {b:>13.6e} {rd:>10.2e} "
                  f"{rd/rel_in:>13.0f}x")

    out = Path(__file__).resolve().parent.parent / "results" / "m1_sensitivity.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    print("\nRESOLUTION FLOOR -- an effect must exceed this to be believable:")
    for mode in MODES:
        worst = max(r["rel_diff"] for r in rows if r["mode"] == mode)
        med = sorted(r["rel_diff"] for r in rows if r["mode"] == mode)[len(BIT_GRID) // 2]
        print(f"  {mode:>4}: median {med:>8.1%}   worst {worst:>8.1%}")
    print("\nAny SR/EF comparison with a margin below its mode's floor is NOT")
    print("resolved by this experiment, regardless of seed count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
