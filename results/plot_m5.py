"""Plot the ablation ladder.  [restored after the data loss]

Three panels, because three different questions were asked of the same runs:

  1. validation loss vs TOKENS, one curve per rung, seeds as a band.
     Tokens, not steps and not wall-clock -- the whole project turns on the fact
     that two token budgets give opposite answers, so the x-axis has to be the
     budget.
  2. final perplexity vs rung, seeds as points, so the spread is visible next to
     the mean instead of being summarized away.
  3. cost vs benefit: ppl penalty against the M4 audited fp_mul that the rung
     removes. This is the panel the project exists to produce, and the one that
     shows the ladder is upside down.

Usage:
    python results/plot_m5.py [--results results/gpu2] [--out results/figs]
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_config(rows: list[dict], key: str) -> dict[str, list[float]]:
    out = collections.defaultdict(list)
    for r in rows:
        try:
            v = float(r[key])
        except (ValueError, KeyError):
            continue
        if math.isfinite(v):
            out[r["config"]].append(v)
    return out


def panel_curves(ax, curves: list[dict]) -> None:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in curves:
        per[r["config"]][int(r["tokens"])].append(float(r["val_loss"]))
    for name, d in sorted(per.items()):
        xs = sorted(d)
        mean = [sum(d[x]) / len(d[x]) for x in xs]
        line, = ax.plot(xs, mean, label=name, lw=1.4)
        # seed spread as a band, never as an error bar: with 3 seeds the min-max
        # IS the data, and a standard error would imply a distribution we have
        # no right to assume
        lo = [min(d[x]) for x in xs]
        hi = [max(d[x]) for x in xs]
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("tokens")
    ax.set_ylabel("validation loss (nats)")
    ax.set_title("loss vs token budget")
    ax.legend(fontsize=6, ncol=2)


def panel_final(ax, rows: list[dict]) -> None:
    ppl = by_config(rows, "val_ppl")
    names = sorted(ppl, key=lambda n: sum(ppl[n]) / len(ppl[n]))
    for i, n in enumerate(names):
        vs = ppl[n]
        ax.scatter([i] * len(vs), vs, s=14, zorder=3)
        ax.hlines(sum(vs) / len(vs), i - 0.3, i + 0.3, lw=2, zorder=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("final validation perplexity")
    ax.set_title("final perplexity -- every seed shown, bar = mean")


def panel_cost_benefit(ax, rows: list[dict], audit: list[dict]) -> None:
    """The panel the project exists to produce."""
    ppl = by_config(rows, "val_ppl")
    fp = {r["config"]: int(r["fp_mul"]) for r in audit}
    if "R0_fp32" not in ppl or "R0_fp32" not in fp:
        ax.set_title("cost vs benefit (needs an R0_fp32 row in both files)")
        return
    base_ppl = sum(ppl["R0_fp32"]) / len(ppl["R0_fp32"])
    base_fp = fp["R0_fp32"]
    for n in sorted(set(ppl) & set(fp)):
        if n == "R0_fp32":
            continue
        cost = (sum(ppl[n]) / len(ppl[n])) / base_ppl - 1.0
        benefit = 1.0 - fp[n] / base_fp
        ax.scatter(benefit * 100, cost * 100, s=30, zorder=3)
        ax.annotate(n, (benefit * 100, cost * 100), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, lw=0.8, color="k")
    ax.set_xlabel("FP multiplies removed (%, M4 audit)")
    ax.set_ylabel("perplexity penalty (%)")
    ax.set_title("cost vs benefit -- down and right is good")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "gpu2"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figs"))
    args = ap.parse_args()

    rdir = Path(args.results)
    rows = read(rdir / "m5_ladder.csv")
    curves = read(rdir / "m5_ladder_curves.csv")
    audit = read(ROOT / "results" / "m4_audit.csv")
    if not rows:
        print(f"no ladder results in {rdir}")
        return 1

    budgets = sorted({int(r["tokens"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panel_curves(axes[0], curves)
    panel_final(axes[1], rows)
    panel_cost_benefit(axes[2], rows, audit)
    fig.suptitle(f"M5 ablation ladder -- {rdir.name}, "
                 f"budget(s) {', '.join(f'{b/1e6:.2f}M' for b in budgets)} tokens",
                 fontsize=10)
    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"m5_{rdir.name}.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}  ({len(rows)} runs, {len(set(r['config'] for r in rows))} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
