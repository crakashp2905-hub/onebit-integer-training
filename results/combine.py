"""Merge every 20.48M-token ladder run into one table.

The runs are spread across three directories because they were produced on
different machines and in different sessions (results/ is local CPU, gpu2/ and
gpu3/ are Kaggle). Merging them is only legitimate because the devices were
checked against each other rather than assumed equivalent: R0_fp32 measured
7.646 on CPU and 7.638 on GPU, a 0.1% gap against a seed spread of 7.598-7.687.

Every rung is still normalized against the R0 baseline measured on ITS OWN
device, so a device offset could not masquerade as a rung effect even if one
appeared later.

    python results/combine.py    ->  results/m5_all_20M.csv
"""

from __future__ import annotations

import collections
import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 20_480_000

#: directory -> device label. The label is what the normalization keys on.
SOURCES = {".": "CPU", "gpu": "GPU", "gpu2": "GPU", "gpu3": "GPU", "gpu4": "GPU"}

#: ladder order, not alphabetical and not sorted by result
ORDER = [
    "R0_fp32", "R1_ternary", "R2_act8", "R2p5_pow2scales",
    "R2p9_act4", "R2p95_act4_pow2", "R3a_dgrad8", "R3b_wgrad8",
    "R6_attn8", "R6p5_attn8_pow2", "R6_attn4", "R7_head8",
    "R7_everything_pow2", "R5_shadow8ef", "R5_shadow8sr", "R5_shadow8rtn",
    "R5_shadow4ef", "S5_shadow8ef_solo",
]


def load() -> list[dict]:
    seen: set[tuple] = set()
    rows = []
    for d, dev in SOURCES.items():
        p = ROOT / "results" / d / "m5_ladder.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["tokens"]) != BUDGET:
                    continue
                key = (r["config"], r["seed"], dev)
                if key in seen:       # gpu2 is a superset of gpu
                    continue
                seen.add(key)
                r["device"] = dev
                r["source"] = d
                rows.append(r)
    return rows


def main() -> int:
    rows = load()
    if not rows:
        print("no 20.48M-token rows found")
        return 1

    audit = {}
    ap = ROOT / "results" / "m4_audit.csv"
    if ap.exists():
        with ap.open(encoding="utf-8") as f:
            audit = {r["config"]: int(r["fp_mul"]) for r in csv.DictReader(f)}
    a0 = audit.get("R0_fp32")

    agg: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        agg[r["config"]][r["device"]].append(float(r["val_ppl"]))
    base = {d: st.mean(v) for d, v in agg["R0_fp32"].items()}

    out = ROOT / "results" / "m5_all_20M.csv"
    fields = ["config", "device", "n_seeds", "ppl_mean", "ppl_min", "ppl_max",
              "pct_vs_fp32", "fp_mul", "fp_mul_pct_of_fp32", "sources"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        names = [c for c in ORDER if c in agg] + \
                sorted(c for c in agg if c not in ORDER)
        for c in names:
            for dev, v in sorted(agg[c].items()):
                srcs = sorted({r["source"] for r in rows
                               if r["config"] == c and r["device"] == dev})
                w.writerow({
                    "config": c, "device": dev, "n_seeds": len(v),
                    "ppl_mean": round(st.mean(v), 4),
                    "ppl_min": round(min(v), 4), "ppl_max": round(max(v), 4),
                    "pct_vs_fp32": round((st.mean(v) / base[dev] - 1) * 100, 2),
                    "fp_mul": audit.get(c, ""),
                    "fp_mul_pct_of_fp32": (round(audit[c] / a0 * 100, 2)
                                           if c in audit and a0 else ""),
                    "sources": "+".join(srcs),
                })

    print(f"{'config':<22} {'dev':<4} {'n':>2} {'ppl':>7} {'vs fp32':>8} "
          f"{'fp_mul':>8}")
    print("-" * 60)
    with out.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            print(f"{r['config']:<22} {r['device']:<4} {r['n_seeds']:>2} "
                  f"{float(r['ppl_mean']):>7.3f} "
                  f"{float(r['pct_vs_fp32']):>+7.1f}% "
                  f"{(r['fp_mul_pct_of_fp32'] or '-'):>7}%")
    print(f"\nwrote {out}  ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
