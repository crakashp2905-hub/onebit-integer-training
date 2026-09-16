"""Fetch a finished kernel's output into results/<dir>/.

Lands in its OWN directory (results/gpu, gpu2, gpu3, ...) rather than merging
into results/m5_ladder.csv. GPU and CPU runs are different devices, and a rung
compared against a baseline measured on the other device confounds the rung
with the arithmetic order. Keeping them apart makes that mistake require
deliberate effort.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default=None, help="user/kernel-slug")
    ap.add_argument("--into", default=None,
                    help="results subdirectory (default: next free gpuN)")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    ref = args.kernel or f"{api.config_values['username']}/onebit-ladder-gpu"

    if args.into:
        out = ROOT / "results" / args.into
    else:
        n = 1
        while (ROOT / "results" / (f"gpu{n}" if n > 1 else "gpu")).exists():
            n += 1
        out = ROOT / "results" / (f"gpu{n}" if n > 1 else "gpu")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[pull] {ref} -> {out}")
    api.kernels_output(ref, path=str(out))

    csvs = sorted(out.glob("*.csv"))
    if not csvs:
        print("[pull] NO CSV in the output. The kernel ran but produced nothing -- "
              "check the log before recording anything from this run.")
        return 1
    for c in csvs:
        with c.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        budgets = sorted({r.get("tokens", "?") for r in rows})
        print(f"  {c.name}: {len(rows)} rows, "
              f"{len({r.get('config') for r in rows})} configs, tokens={budgets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
