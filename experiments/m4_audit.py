"""M4: the float-op census. What did the ladder actually eliminate?

This was in the original build order between M3 and M5 and got skipped. Without
it the project can report what quantization COSTS (loss numbers) but not what it
BOUGHT, and "multiplier-free" is a claim about operations, not about loss.

Emits, for each ladder config, the number of floating-point multiplies in ONE
training step, broken down by source, plus the irreducible residue: the
transcendentals (exp, rsqrt, erf) that no scaling scheme removes.

Reading the numbers:

  fp_mul  what a REAL integer kernel would still have to do in floating point.
          A ternary matmul contributes 0 (multiplying by {-1,0,+1} is a
          select/negate/skip); its requantization scale contributes one FP
          multiply per output element unless that scale is a power of two.
  raw_mul what Track A literally executes. Simulated quantization runs on FP
          units, so this barely moves -- which is exactly why fp_mul has to be
          modelled rather than measured.

Output: results/m4_audit.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from audit import FloatOpCounter, attach_labels
from experiments.repro import pin
from models import GPTConfig, QuantGPT
from models.quant_mlp import LadderConfig
from optim import IntSGD
from quant import QuantSpec

BATCH, CTX = 4, 128

TERN = QuantSpec(kind="ternary", granularity="tensor", calib="absmean")
TERN_P2 = QuantSpec(kind="ternary", granularity="tensor", calib="absmean",
                    scale_mode="pow2", pow2_mode="ceil")
ACT8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")
ACT8_P2 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax",
                    scale_mode="pow2", pow2_mode="ceil")
ACT4 = QuantSpec(kind="int", bits=4, granularity="row", calib="absmax")
ACT4_P2 = QuantSpec(kind="int", bits=4, granularity="row", calib="absmax",
                    scale_mode="pow2", pow2_mode="ceil")
ATT8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")
ATT8_P2 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax",
                    scale_mode="pow2", pow2_mode="ceil")
ATT4 = QuantSpec(kind="int", bits=4, granularity="row", calib="absmax")
HEAD8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")
HEAD8_P2 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax",
                     scale_mode="pow2", pow2_mode="ceil")
DY8 = QuantSpec(kind="int", bits=8, granularity="row", rounding="stochastic")
WG8 = QuantSpec(kind="int", bits=8, granularity="tensor", rounding="stochastic")


def configs() -> list[LadderConfig]:
    full = dict(w_spec=TERN, a_spec=ACT8, g_spec=DY8, wg_spec=WG8)
    return [
        LadderConfig(name="R0_fp32"),
        LadderConfig(name="R1_ternary", w_spec=TERN),
        LadderConfig(name="R2_act8", w_spec=TERN, a_spec=ACT8),
        LadderConfig(name="R2p5_pow2scales", w_spec=TERN_P2, a_spec=ACT8_P2),
        LadderConfig(name="R2p9_act4", w_spec=TERN, a_spec=ACT4),
        LadderConfig(name="R3b_wgrad8", **full),
        # --- 4-bit multiplier-free: M0 predicts pow2 scaling bites HERE, not at 8
        LadderConfig(name="R2p95_act4_pow2", w_spec=TERN_P2, a_spec=ACT4_P2),
        # --- R6: the two activation x activation matmuls inside attention
        LadderConfig(name="R6_attn8", **full, attn_spec=ATT8),
        LadderConfig(name="R6p5_attn8_pow2", **full, attn_spec=ATT8_P2,
                     attn_scale_pow2=True),
        LadderConfig(name="R6_attn4", **full, attn_spec=ATT4),
        # --- R7: the LM head, 55% of the residue and never before in the ladder
        LadderConfig(name="R7_head8", **full, attn_spec=ATT8, head_spec=HEAD8),
        LadderConfig(name="R7_everything_pow2", w_spec=TERN_P2, a_spec=ACT8_P2,
                     g_spec=DY8, wg_spec=WG8, attn_spec=ATT8_P2,
                     attn_scale_pow2=True, head_spec=HEAD8_P2),
        LadderConfig(name="R5_shadow8ef", **full, shadow_bits=8, shadow_mode="ef"),
        LadderConfig(name="R7_alllayers", **full, quantize_first_last=True),
    ]


def audit_one(cfg: LadderConfig) -> dict:
    torch.manual_seed(0)
    gcfg = GPTConfig(vocab_size=2048, ctx=CTX, n_layer=6, n_head=6, d_model=192,
                     ladder=cfg)
    model = QuantGPT(gcfg, seed=0)
    shadow = (QuantSpec(kind="int", bits=cfg.shadow_bits, granularity="tensor",
                        rounding="nearest") if cfg.shadow_bits else None)
    opt = IntSGD(model.parameters(), lr=0.5, momentum=0.9, shadow_spec=shadow,
                 shadow_range="auto" if cfg.shadow_bits else None,
                 grad_spec=cfg.wg_spec if cfg.wg_spec.kind != "none" else None,
                 error_feedback=(cfg.shadow_mode == "ef" and bool(cfg.shadow_bits)),
                 seed=0)

    x = torch.randint(0, 2048, (BATCH, CTX))
    y = torch.randint(0, 2048, (BATCH, CTX))

    counter = FloatOpCounter()
    handles = attach_labels(model, counter)
    with counter:
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
    for h in handles:
        h.remove()

    t = counter.totals()
    by = counter.by_source()

    def group(pred) -> int:
        return sum(b.fp_mul for n, b in by if pred(n))

    return {
        "config": cfg.name,
        "spec": cfg.describe(),
        "fp_mul": t["fp_mul"],
        "raw_mul": t["raw_mul"],
        "transcendental": t["transcendental"],
        "add": t["add"],
        "mul_block_linears": group(lambda n: "QuantLinear" in n),
        "mul_attention": group(lambda n: "CausalSelfAttention" in n),
        "mul_layernorm": group(lambda n: "LayerNorm" in n),
        "mul_head_embed": group(lambda n: "Linear]" in n and "Quant" not in n
                                or "Embedding" in n),
        "mul_other": group(lambda n: not any(k in n for k in
                                             ("QuantLinear", "CausalSelfAttention",
                                              "LayerNorm", "Embedding"))
                           and "Linear]" not in n),
        "_by": by,
    }


def main() -> int:
    pin()
    rows = [audit_one(c) for c in configs()]
    out = Path(__file__).resolve().parent.parent / "results" / "m4_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if not k.startswith("_")]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    base = rows[0]["fp_mul"]
    print(f"one training step, batch {BATCH} x ctx {CTX}, 3.09M-param decoder")
    print("fp_mul = FP multiplies a REAL integer kernel would still perform\n")
    print(f"{'config':<20} {'fp_mul':>15} {'vs fp32':>9} {'transcend':>12} {'raw_mul':>15}")
    print("-" * 76)
    for r in rows:
        print(f"{r['config']:<20} {r['fp_mul']:>15,} {r['fp_mul']/base:>8.1%} "
              f"{r['transcendental']:>12,} {r['raw_mul']:>15,}")

    print(f"\nwhere the SURVIVING multiplies live (config: {rows[-2]['config']}):")
    for name, b in rows[-2]["_by"][:10]:
        if b.fp_mul:
            print(f"  {name:<46} {b.fp_mul:>13,}")

    print("\nirreducible residue -- transcendentals no scaling scheme removes:")
    print(f"  {rows[-1]['transcendental']:,} per step "
          f"(softmax exp/divide, LayerNorm rsqrt, GELU erf)")
    print(f"  = {rows[-1]['transcendental'] / (BATCH * CTX):,.0f} per token")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
