"""Are the M1 findings spectrum-dependent?

M1 measured everything on a log-spaced eigenspectrum. That is a convenient
choice, not a realistic one: measured neural-network Hessians are dominated by a
near-zero bulk with a few large outliers. Before any M1 conclusion is used as
guidance for R5, it has to survive a change of spectrum.

Findings under test (from results/RESULTS.md, M1 section):

  F1  RTN wins on final loss IFF it freezes completely            (12/12 on logspace)
  F2  EF is 2-2.5x tighter than SR in the infinity norm
  F3  error geometry: RTN piles error into soft directions,
      EF into stiff ones, SR spreads it

Any claim that flips under a spectrum change is a claim about log-spaced
quadratics, not about quantized optimization.

Output: results/m1_spectrum.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments.convex_problem import make_quadratic
from experiments.m1_convex import MODES, run
from experiments.repro import pin, threads

D = 1000
STEPS = 3000
KAPPA = 100.0
BITS = [16, 10, 8, 4]
SEEDS = (0, 1, 2)
SPECTRA = ("logspace", "powerlaw", "bulk_outlier")

# measured in results/m1_sensitivity.csv -- effects below this are not resolved
FLOOR = {"rtn": 0.214, "sr": 0.262, "ef": 0.344}


def main() -> int:
    n = pin()
    print(f"[repro] torch threads pinned to {n}\n")
    rows = []
    for spectrum in SPECTRA:
        p = make_quadratic(D, KAPPA, seed=0, spectrum=spectrum)
        for bits in BITS:
            for mode in MODES:
                for seed in SEEDS:
                    r = run(p, bits=bits, mode=mode, steps=STEPS, seed=seed,
                            check_freeze_law=(mode == "rtn"))
                    r["spectrum"] = spectrum
                    rows.append(r)
        print(f"  {spectrum} done ({len(rows)} runs)")

    fields = ["spectrum"] + [k for k in rows[0] if not k.startswith("_") and k != "spectrum"]
    out = Path(__file__).resolve().parent.parent / "results" / "m1_spectrum.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  wrote {out} ({len(rows)} rows)\n")

    def mean(spectrum, bits, mode, key):
        v = [r[key] for r in rows
             if r["spectrum"] == spectrum and r["bits"] == bits and r["mode"] == mode]
        return sum(v) / len(v)

    viol = sum(r["freeze_law_violations"] for r in rows
               if isinstance(r["freeze_law_violations"], int))
    print(f"L1 freeze law across all spectra: {viol} violations "
          f"-> {'HOLDS' if viol == 0 else 'VIOLATED'}\n")

    print("=" * 78)
    print("F1 -- does 'RTN wins IFF it fully freezes' survive?")
    print("=" * 78)
    print(f"  {'spectrum':>13} {'bits':>4} {'RTNfroz':>8} {'RTN':>10} {'SR':>10} "
          f"{'EF':>10} {'winner':>7} {'F1':>5}")
    ok = tot = 0
    for spectrum in SPECTRA:
        for bits in BITS:
            g = {m: mean(spectrum, bits, m, "rel_gap") for m in MODES}
            froz = mean(spectrum, bits, "rtn", "frozen_frac_tail")
            winner = min(g, key=g.get)
            holds = (winner == "rtn") == (froz == 1.0)
            ok += holds; tot += 1
            print(f"  {spectrum:>13} {bits:>4} {froz:>8.3f} {g['rtn']:>10.2e} "
                  f"{g['sr']:>10.2e} {g['ef']:>10.2e} {winner:>7} "
                  f"{'OK' if holds else 'FAIL':>5}")
    print(f"\n  F1 holds in {ok}/{tot} cells")

    print()
    print("=" * 78)
    print("F2 -- is EF still tighter than SR in the infinity norm?")
    print("=" * 78)
    print(f"  {'spectrum':>13} {'bits':>4} {'SR steps':>9} {'EF steps':>9} {'ratio':>7} "
          f"{'margin':>8} {'resolved':>9}")
    for spectrum in SPECTRA:
        for bits in BITS:
            sr = mean(spectrum, bits, "sr", "err_in_grid_steps")
            ef = mean(spectrum, bits, "ef", "err_in_grid_steps")
            margin = abs(sr - ef) / max(sr, ef)
            print(f"  {spectrum:>13} {bits:>4} {sr:>9.2f} {ef:>9.2f} {sr/ef:>7.2f} "
                  f"{margin:>7.1%} {'yes' if margin > FLOOR['ef'] else 'NO':>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
