"""M1: can the integer optimizer descend a paraboloid, and where does it stop?

Not only a sanity gate. A convex quadratic has an exactly known gradient at every
point, so M1 doubles as a calibrated instrument for R5: predictions about
quantized latent weights can be CHECKED here rather than assumed, in seconds
rather than in a training run.

Laws under test (zero free parameters):

  L1  FREEZE LAW (exact). Under round-to-nearest with grid step D, a coordinate
      already on the grid satisfies  Q(w_i - lr*g_i) = w_i  <=>  |lr*g_i| < D/2.
      So: every frozen coordinate must have |lr*g_i| < D/2, and every moving one
      must have |lr*g_i| >= D/2. Violations must be exactly ZERO. This is a
      correctness check on the whole stack -- optimizer, quantizer, and problem.

  L2  Whether RTN freezes COMPLETELY depends on the grid step relative to the
      curvature spectrum, not on bit-width alone. Measured, not asserted.

  L3  EF tracks exact GD to within a bounded residual, so its final distance from
      the optimum should be a small multiple of D, and should shrink
      proportionally to D as bits increase.

  L4  SR is unbiased so it keeps moving, but injects variance every step, so it
      should sit in a noise ball rather than converge.

  L5  ERROR GEOMETRY. The three modes differ in WHERE they put their error, not
      only how much. Measured by projecting (w - w*) onto the eigenbasis of A:
      RTN piles error into soft directions (they freeze first, far out), EF into
      stiff directions (they limit-cycle at the residual bound), SR spreads it.
      Since the objective is the A-norm, this is why EF's much smaller inf-norm
      error buys it nothing over SR on the loss.

Outputs:
  results/m1_convex.csv         one row per (kappa, bits, mode, seed)
  results/m1_convex_curves.csv  loss curves for plotting
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments.repro import pin, threads
from experiments.convex_problem import Quadratic, make_quadratic, verify_problem
from optim import IntSGD
from quant import QuantSpec

D = 1000
STEPS = 3000
SHADOW_RANGE = 8.0
GATE_REL_GAP = 1e-3
BIT_GRID = [16, 12, 10, 8, 6, 4]
KAPPAS = (10.0, 100.0)
SEEDS = (0, 1, 2)

MODES = {
    "rtn": dict(rounding="nearest", error_feedback=False),
    "sr": dict(rounding="stochastic", error_feedback=False),
    "ef": dict(rounding="nearest", error_feedback=True),
}


def run(
    p: Quadratic,
    *,
    bits: int | None,
    mode: str = "rtn",
    steps: int = STEPS,
    lr: float | None = None,
    seed: int = 0,
    grad_bits: int | None = None,
    record_every: int = 25,
    check_freeze_law: bool = True,
) -> dict:
    """One optimization run. bits=None means an unquantized FP32 shadow."""
    lr = p.lr_opt if lr is None else lr
    w = torch.zeros(p.d, requires_grad=True)
    w0 = w.detach().clone()

    if bits is None:
        shadow_spec, shadow_range, ef = QuantSpec(kind="none"), None, False
    else:
        cfg = MODES[mode]
        shadow_spec = QuantSpec(
            kind="int", bits=bits, granularity="tensor", rounding=cfg["rounding"]
        )
        shadow_range = SHADOW_RANGE
        ef = cfg["error_feedback"]

    grad_spec = (
        QuantSpec(kind="int", bits=grad_bits, granularity="tensor", rounding="stochastic")
        if grad_bits
        else None
    )

    opt = IntSGD(
        [w],
        lr=lr,
        shadow_spec=shadow_spec,
        shadow_range=shadow_range,
        grad_spec=grad_spec,
        error_feedback=ef,
        seed=seed,
    )
    delta = opt.delta

    curve: list[tuple[int, float]] = []
    frozen_tail: list[float] = []
    sat_max = 0.0
    freeze_violations = 0
    best = float("inf")
    t0 = time.time()

    # L1 is only meaningful for plain round-to-nearest: SR is random and EF adds
    # a residual, so neither obeys the deterministic freeze law.
    verify_law = check_freeze_law and bits is not None and mode == "rtn"

    for t in range(steps):
        g = p.grad(w.detach())
        w.grad = g
        w_before = w.detach().clone() if verify_law and t >= steps - 50 else None
        opt.step()
        sat_max = max(sat_max, opt.last["shadow_sat_frac"])

        if w_before is not None:
            frozen = w.detach() == w_before
            step_mag = (lr * g).abs()
            # frozen  =>  |lr*g| <  D/2      moving  =>  |lr*g| >= D/2
            freeze_violations += int((step_mag[frozen] >= delta / 2).sum().item())
            freeze_violations += int((step_mag[~frozen] < delta / 2).sum().item())

        if t % record_every == 0 or t == steps - 1:
            rg = p.rel_gap(w.detach(), w0)
            curve.append((t, rg))
            best = min(best, rg)
        if t >= steps - 200:
            frozen_tail.append(opt.last["frozen_frac"])

    w_fin = w.detach()
    rel_gap = p.rel_gap(w_fin, w0)
    err_inf = (w_fin - p.w_star).abs().max().item()

    return {
        "kappa": p.kappa,
        "bits": bits if bits is not None else "fp32",
        "mode": mode if bits is not None else "-",
        "grad_bits": grad_bits or "",
        "seed": seed,
        "lr": lr,
        "steps": steps,
        "delta": delta if delta is not None else "",
        "rel_gap": rel_gap,
        "best_rel_gap": min(best, rel_gap),
        "passed_gate": rel_gap < GATE_REL_GAP,
        "w_err_inf": err_inf,
        "err_in_grid_steps": err_inf / delta if delta else "",
        "frozen_frac_tail": sum(frozen_tail) / len(frozen_tail) if frozen_tail else 0.0,
        "freeze_law_violations": freeze_violations if verify_law else "",
        "shadow_sat_frac_max": sat_max,
        "threads": threads(),
        "seconds": round(time.time() - t0, 2),
        "_curve": curve,
    }


def check_gd_matches_theory(p: Quadratic, steps: int = 400) -> dict:
    """Unquantized IntSGD must reproduce textbook GD convergence.

    rel_gap is quadratic in the error, so it contracts at rho^2 per step where
    rho = (kappa-1)/(kappa+1). If this fails, nothing downstream can be trusted.
    """
    r = run(p, bits=None, steps=steps, record_every=1, check_freeze_law=False)
    curve = r["_curve"]
    lo, hi = curve[steps // 2], curve[-1]
    n = hi[0] - lo[0]
    measured = (hi[1] / lo[1]) ** (1.0 / n) if lo[1] > 0 and hi[1] > 0 else float("nan")
    return {
        "kappa": p.kappa,
        "rate_measured_per_step": measured,
        "rate_theory_per_step": p.rate_opt**2,
        "final_rel_gap": r["rel_gap"],
    }


def main() -> int:
    n_threads = pin()
    print(f"[repro] torch threads pinned to {n_threads}")
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1 -- verify the problem against its own closed form")
    print("=" * 78)
    for k in (1.0, *KAPPAS):
        v = verify_problem(make_quadratic(D, k, 0))
        ok = (
            abs(v["kappa_measured"] - k) / k < 1e-3
            and v["grad_at_optimum_rel"] < 1e-5
            and v["gap_at_optimum"] < 1e-12
        )
        print(
            f"  kappa={k:<6} measured={v['kappa_measured']:.4f}  "
            f"|grad(w*)|_rel={v['grad_at_optimum_rel']:.1e}  "
            f"gap(w*)={v['gap_at_optimum']:.1e}   {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            print("  problem construction is wrong; stopping")
            return 1

    print()
    print("=" * 78)
    print("STEP 2 -- unquantized run must reproduce textbook GD convergence")
    print("=" * 78)
    print(f"  {'kappa':>6}  {'measured rate/step':>19}  {'theory rho^2':>13}  {'ratio':>7}")
    theory_ok = True
    for k in KAPPAS:
        c = check_gd_matches_theory(make_quadratic(D, k, 0))
        ratio = c["rate_measured_per_step"] / c["rate_theory_per_step"]
        theory_ok &= abs(ratio - 1.0) < 0.02
        print(
            f"  {k:>6.0f}  {c['rate_measured_per_step']:>19.6f}  "
            f"{c['rate_theory_per_step']:>13.6f}  {ratio:>7.4f}"
        )
    print(f"  -> {'matches theory' if theory_ok else 'DOES NOT MATCH THEORY'}")

    print()
    print("=" * 78)
    print("STEP 3 -- the sweep")
    print("=" * 78)
    rows, curves = [], []

    def keep(r):
        rows.append(r)
        for t, v in r["_curve"]:
            curves.append(
                {"kappa": r["kappa"], "bits": r["bits"], "mode": r["mode"],
                 "seed": r["seed"], "step": t, "rel_gap": v}
            )

    t_start = time.time()
    for kappa in KAPPAS:
        p = make_quadratic(D, kappa, seed=0)
        for seed in SEEDS:
            keep(run(p, bits=None, seed=seed, check_freeze_law=False))
        for bits in BIT_GRID:
            for mode in MODES:
                for seed in SEEDS:
                    keep(run(p, bits=bits, mode=mode, seed=seed))
        print(f"  kappa={kappa:.0f} done ({len(rows)} runs, {time.time()-t_start:.0f}s)")

    fields = [k for k in rows[0] if not k.startswith("_")]
    with (out_dir / "m1_convex.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    with (out_dir / "m1_convex_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)
    print(f"  wrote results/m1_convex.csv ({len(rows)} rows)")
    print(f"  wrote results/m1_convex_curves.csv ({len(curves)} rows)")

    def sel(kappa, bits, mode, key):
        return [
            r[key] for r in rows
            if r["kappa"] == kappa and r["bits"] == bits and r["mode"] == mode
        ]

    def mean(kappa, bits, mode, key):
        v = sel(kappa, bits, mode, key)
        return sum(v) / len(v)

    for kappa in KAPPAS:
        print()
        print("-" * 78)
        print(f"kappa = {kappa:.0f}   (mean over {len(SEEDS)} seeds; lr = {2/(kappa+1):.5f})")
        print("-" * 78)
        print(f"  fp32 shadow: final rel_gap = {mean(kappa,'fp32','-','rel_gap'):.3e}")
        print()
        print(f"  {'bits':>4} {'delta':>9} | {'RTN':>10} {'SR':>10} {'EF':>10}  <- final rel_gap")
        for bits in BIT_GRID:
            d = SHADOW_RANGE / (2 ** (bits - 1) - 1)
            print(f"  {bits:>4} {d:>9.2e} | "
                  f"{mean(kappa,bits,'rtn','rel_gap'):>10.2e} "
                  f"{mean(kappa,bits,'sr','rel_gap'):>10.2e} "
                  f"{mean(kappa,bits,'ef','rel_gap'):>10.2e}")
        print()
        print(f"  {'bits':>4} | {'RTN':>10} {'SR':>10} {'EF':>10}  <- final ||w-w*||_inf "
              f"in GRID STEPS")
        for bits in BIT_GRID:
            print(f"  {bits:>4} | "
                  f"{mean(kappa,bits,'rtn','err_in_grid_steps'):>10.1f} "
                  f"{mean(kappa,bits,'sr','err_in_grid_steps'):>10.1f} "
                  f"{mean(kappa,bits,'ef','err_in_grid_steps'):>10.1f}")
        print()
        print(f"  {'bits':>4} | {'RTN':>10} {'SR':>10} {'EF':>10}  <- frozen coord fraction")
        for bits in BIT_GRID:
            print(f"  {bits:>4} | "
                  f"{mean(kappa,bits,'rtn','frozen_frac_tail'):>10.3f} "
                  f"{mean(kappa,bits,'sr','frozen_frac_tail'):>10.3f} "
                  f"{mean(kappa,bits,'ef','frozen_frac_tail'):>10.3f}")

    print()
    print("=" * 78)
    print("STEP 4 -- law checks")
    print("=" * 78)
    viol = sum(
        r["freeze_law_violations"] for r in rows
        if isinstance(r["freeze_law_violations"], int)
    )
    n_checked = sum(1 for r in rows if isinstance(r["freeze_law_violations"], int))
    print(f"  L1 freeze law: {viol} violations across {n_checked} RTN runs "
          f"(50 steps x {D} coords each) -> {'HOLDS' if viol == 0 else 'VIOLATED'}")

    sat = max(r["shadow_sat_frac_max"] for r in rows)
    print(f"  saturation: max shadow_sat_frac = {sat:.2e} "
          f"-> {'no clipping, analysis clean' if sat == 0 else 'CLIPPING, analysis confounded'}")

    print()
    print("  L2 RTN freeze fraction (1.000 = complete stop; depends on grid step")
    print("     vs curvature spectrum, NOT on bit-width alone):")
    for kappa in KAPPAS:
        fr = " ".join(
            f"b{b}={mean(kappa, b, 'rtn', 'frozen_frac_tail'):.3f}"
            for b in sorted(BIT_GRID)
        )
        print(f"    kappa={kappa:>3.0f}  {fr}")

    print()
    print("  L3 EF error vs delta (should scale linearly -- 4x coarser grid, 4x error):")
    for kappa in KAPPAS:
        errs = [(b, mean(kappa, b, "ef", "w_err_inf")) for b in sorted(BIT_GRID)]
        line = f"    kappa={kappa:>3.0f}  " + " ".join(f"b{b}={e:.2e}" for b, e in errs)
        print(line)
        ratios = [
            (errs[i + 1][1] / errs[i][1])
            / ((SHADOW_RANGE / (2 ** (errs[i + 1][0] - 1) - 1))
               / (SHADOW_RANGE / (2 ** (errs[i][0] - 1) - 1)))
            for i in range(len(errs) - 1)
        ]
        print(f"             error-ratio / delta-ratio: "
              + " ".join(f"{r:.2f}" for r in ratios) + "   (1.00 = exactly linear)")

    print()
    print("=" * 78)
    print("STEP 5 -- error GEOMETRY: where does each mode put its error?")
    print("=" * 78)
    report_error_geometry(bits=8)
    return 0


def report_error_geometry(bits: int = 8, kappa: float = 100.0, steps: int = STEPS) -> None:
    """Project the final error onto the eigenbasis of A and bin by curvature.

    Explains why EF can have a much smaller inf-norm error than SR and still not
    beat it on the loss: the objective is the A-norm, which charges stiff
    directions more.
    """
    p = make_quadratic(D, kappa, seed=0)
    lam, V = torch.linalg.eigh(p.A.double())
    bands = [(1, 3), (3, 10), (10, 32), (32, 100)]
    print(f"  kappa={kappa:.0f}, bits={bits}. Fraction of squared error energy per")
    print("  curvature band (eigenvalues log-spaced, so each band holds ~25% of coords).")
    print()
    header = " ".join(f"lam {a}-{b}".rjust(11) for a, b in bands)
    print(f"  {'mode':>4} {'|w-w*|_inf':>11} {'A-norm err':>11} | {header}")
    for mode in MODES:
        cfg = MODES[mode]
        w = torch.zeros(p.d, requires_grad=True)
        opt = IntSGD(
            [w],
            lr=p.lr_opt,
            shadow_spec=QuantSpec(
                kind="int", bits=bits, granularity="tensor", rounding=cfg["rounding"]
            ),
            shadow_range=SHADOW_RANGE,
            error_feedback=cfg["error_feedback"],
            seed=0,
        )
        for _ in range(steps):
            w.grad = p.grad(w.detach())
            opt.step()
        dv = (w.detach() - p.w_star).double()
        e2 = (V.T @ dv) ** 2
        anorm = float((lam * e2).sum().sqrt())
        fr = [float(e2[(lam >= a) & (lam < b)].sum() / e2.sum()) for a, b in bands]
        print(f"  {mode:>4} {dv.abs().max():>11.3e} {anorm:>11.3e} | "
              + " ".join(f"{f:>11.3f}" for f in fr))
    print()
    print("  Flat row = isotropic error. Right-heavy = error in stiff directions,")
    print("  which the A-norm objective charges the most for.")


if __name__ == "__main__":
    raise SystemExit(main())
