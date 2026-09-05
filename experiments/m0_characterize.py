"""M0 characterization: static quantization error across the spec space.

SCOPE WARNING. This measures the error a quantizer makes on ONE tensor, in
isolation. It is NOT a training result and says nothing directly about whether
training converges. It exists to (a) confirm the primitives behave the way the
theory says, and (b) get an early, cheap read on one specific question:

    Does constraining the scale to a power of two (multiplier-free) cost more
    than dropping a bit of width?

That question is the R1.5 / R2.5 rung. Answering it here in seconds is much
cheaper than answering it in a training run, and it tells us whether to expect
the scale-representation axis or the bit-width axis to bind first.

Output: results/m0_quantizer_error.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from experiments.repro import pin, threads
from quant import QuantSpec, fake_quant, make_generator

SEED = 0
SHAPE = (256, 1024)


def nrmse(x: torch.Tensor, q: torch.Tensor) -> float:
    """Quantization error relative to signal magnitude. Scale-free, so numbers
    are comparable across distributions."""
    return (q - x).pow(2).mean().sqrt().item() / x.pow(2).mean().sqrt().item()


def distributions(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    gauss = torch.randn(SHAPE, generator=g)
    # heavy-tailed: what activations actually look like once a few channels blow up
    heavy = gauss.clone()
    idx = torch.randint(0, SHAPE[1], (8,), generator=g)
    heavy[:, idx] *= 30.0
    return {"gaussian": gauss, "outlier_channels": heavy}


def specs() -> list[QuantSpec]:
    out: list[QuantSpec] = []
    for bits in (8, 7, 6, 5, 4, 3, 2):
        for gran, bs in (("tensor", 0), ("row", 0), ("block", 64)):
            for smode, p2, mant in (
                ("float", "ceil", 0),
                ("pow2", "ceil", 0),
                ("pow2", "round", 0),
                ("dyadic", "ceil", 2),
                ("dyadic", "ceil", 4),
            ):
                out.append(
                    QuantSpec(
                        kind="int",
                        bits=bits,
                        granularity=gran,
                        block_size=bs,
                        scale_mode=smode,
                        pow2_mode=p2,
                        mantissa_bits=mant,
                    )
                )
    for gran, bs in (("tensor", 0), ("row", 0), ("block", 64)):
        for smode in ("float", "pow2"):
            out.append(
                QuantSpec(
                    kind="ternary",
                    granularity=gran,
                    block_size=bs,
                    calib="absmean",
                    scale_mode=smode,
                )
            )
    return out


def main() -> int:
    n_threads = pin()
    print(f"[repro] torch threads pinned to {n_threads}")
    rows = []
    for dist_name, x in distributions(SEED).items():
        for spec in specs():
            g = make_generator(1)
            q = fake_quant(x, spec=spec, generator=g)
            rows.append(
                {
                    "distribution": dist_name,
                    "spec": spec.describe(),
                    "kind": spec.kind,
                    "bits": spec.bits if spec.kind == "int" else 1.58,
                    "granularity": spec.granularity
                    + (str(spec.block_size) if spec.granularity == "block" else ""),
                    "scale_mode": spec.scale_mode
                    + ("/" + spec.pow2_mode if spec.scale_mode == "pow2" else "")
                    + ("/m" + str(spec.mantissa_bits) if spec.scale_mode == "dyadic" else ""),
                    "nrmse": round(nrmse(x, q), 6),
                }
            )

    out = Path(__file__).resolve().parent.parent / "results" / "m0_quantizer_error.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, f"({len(rows)} rows)")

    # ---- the one comparison this script exists to make -----------------------
    def look(dist, bits, gran, smode):
        for r in rows:
            if (
                r["distribution"] == dist
                and r["bits"] == bits
                and r["granularity"] == gran
                and r["scale_mode"] == smode
            ):
                return r["nrmse"]
        raise KeyError((dist, bits, gran, smode))

    print("\nIs power-of-two scaling more expensive than one bit of width?")
    print("(NRMSE, gaussian, per-row scaling; lower is better)\n")
    print(f"{'bits':>5}  {'float scale':>12}  {'pow2/ceil':>12}  {'pow2/round':>12}  "
          f"{'float @ bits-1':>15}")
    for bits in (8, 6, 4, 3):
        fl = look("gaussian", bits, "row", "float")
        pc = look("gaussian", bits, "row", "pow2/ceil")
        pr = look("gaussian", bits, "row", "pow2/round")
        lo = look("gaussian", bits - 1, "row", "float") if bits - 1 >= 2 else float("nan")
        print(f"{bits:>5}  {fl:>12.5f}  {pc:>12.5f}  {pr:>12.5f}  {lo:>15.5f}")

    print("\nGranularity vs outliers (int8, float scale):")
    for gran in ("tensor", "row", "block64"):
        print(f"  {gran:<9} gaussian {look('gaussian', 8, gran, 'float'):.5f}   "
              f"outliers {look('outlier_channels', 8, gran, 'float'):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
