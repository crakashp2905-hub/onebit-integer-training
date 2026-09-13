# The ablation ladder

*[restored after the data loss, and extended with R6/R7 and the M4 audit column]*

A rung is a single, named, independently-toggleable place where a floating-point
number is replaced by a low-precision one. The ladder is cumulative by default —
each rung is added on top of the last — with **solo controls** where the
cumulative result would be ambiguous.

Everything here is **Track A: simulated quantization in pure PyTorch**. An INT8
matmul still executes as an FP32 matmul. There is no speedup and there is not
supposed to be one. The deliverable is where and why training breaks, plus the
M4 audit's answer to what each rung *would* buy in a real integer kernel.

---

## The four components, and where each one actually lives

The brief named four numerical components. Two of them turn out not to live
where you would guess, which is the first thing worth writing down.

| component | rung | lives in |
|---|---|---|
| weights | R1 | `QuantLinear.forward` — quantized per forward pass, not stored |
| activations | R2 | `QuantLinear.forward` — the input, per-token |
| gradients (input) | R3a | `quantize_backward` on the layer *output* — identity forward, quantizes `dy` on backward |
| gradients (weight) | R3b | the **optimizer**, because that is where the accumulated weight gradient is |
| optimizer state | R4 | the optimizer — **not built**; IntSGD still carries FP32 momentum |
| latent/shadow weights | R5 | the **optimizer**, because that is where the master copy is |

R3b and R5 living in the optimizer is not a detail. It is why
`registry.assert_no_noops()` — which only sees model-side sites — could not
guard R5, and why `IntSGD.assert_quantizers_active()` had to be written
separately. A rung that is "enabled" but silently inert produces a beautiful,
meaningless FP32 curve, and that is the single most likely way this project
dies quietly.

---

## The rungs

| rung | what it quantizes | notes |
|---|---|---|
| **R0** | nothing | FP32 control. Must be a plain `Linear`, asserted by test. |
| **R1** | weights → ternary | BitNet b1.58 absmean: β = mean\|W\|, codes = clamp(round(W/β), −1, 1) |
| **R2** | activations → INT8 | per-token (row) absmax |
| **R2.5** | R2 with **pow2 scales** | the multiplier-free probe: every scale a shift |
| **R2.9** | activations → INT4 | bit-width, holding scale representation fixed |
| **R2.95** | R2.9 with **pow2 scales** | 4-bit × multiplier-free, where M0 predicts the constraint bites |
| **R3a** | dgrad (`dy`) → INT8 | stochastic rounding; covers the dynamic operand of *both* backward matmuls |
| **R3b** | wgrad → INT8 | stochastic rounding, in the optimizer |
| **R4** | optimizer state | **not built** |
| **R5** | latent weight → INT8/INT4 | three modes: `rtn`, `sr` (stochastic), `ef` (error feedback). The crux rung. |
| **R6** | QK^T and AV operands | q, k, v and the post-softmax probabilities |
| **R6.5** | R6 + pow2 attention scale | 1/√head_dim forced to a shift |
| **R7** | the LM head matmul | 55% of the residue; was an `nn.Linear` outside the ladder entirely |

### What is deliberately *not* a rung

- **Embeddings.** A gather is not a multiply. Quantizing the table would change
  the model without removing arithmetic.
- **Gated MLPs.** The MLP is deliberately ungated: a gated MLP multiplies two
  activations, which is a data-dependent multiply no scaling scheme turns into a
  shift. Keeping the target honest starts at the architecture.
- **`scaled_dot_product_attention`.** Attention is written out longhand so QK^T
  and AV stay visible, quantizable, and countable. SDPA would fuse them into a
  kernel where R6 is impossible and the M4 audit is blind.

---

## R5 is the crux, and why

R5 is the only rung where the quantization grid interacts with the *update*
rather than with a single forward pass. That produces **swamping**: an update
smaller than half a grid step is annihilated, permanently.

The **freeze law** states this exactly, per coordinate:

> coordinate *i* is frozen ⟺ |lr · gᵢ| < Δ/2

Measured: **0 violations across 1.8M coordinate-checks** on three curvature
spectra. It predicted the M2 MNIST collapse quantitatively *before* that run
(Δ = 0.00394 needs |g| < 0.0197, actual ≈ 1e−3 → total freeze; measured
`frozen_frac` = 0.999, loss 2.2984 against ln(10) = 2.3026).

An earlier **aggregate** version of this law — ‖g‖∞ < Δ/(2·lr) — was wrong by
4.3×. The per-coordinate statement is the one that holds; the aggregate one
generalized from an instrument that could not support it.

---

## R6 is structurally different from every other rung

Every rung from R1 to R5 has a **weight** on one side of the matmul, and a
ternary weight makes a matmul multiplier-free on its own, whatever the other
operand is: multiplying by {−1, 0, +1} is a select/negate/skip. That is the
BitNet insight and it is why R1 alone already claims the forward pass.

R6 has no weight. QK^T and AV multiply two **activations**, both dynamic, both
data-dependent. Both operands must be integer or neither matmul is. This is the
one place the BitNet shortcut does not apply, and it is why attention is the
hard part of "multiplier-free" rather than an afterthought.

Two honest caveats recorded up front:

- Post-softmax probabilities are **non-negative**, so a symmetric signed
  quantizer spends a whole bit on a sign that never occurs: 8-bit attention
  probabilities are really 7-bit. Recorded, not fixed — fixing it would confound
  this rung with a zero-point change.
- R6 quantizes the attention **forward** only. dQ/dK/dV are still FP, and the
  M4 audit scores them as such, so R6's measured benefit understates what a full
  treatment would give.

---

## Methodology, non-negotiable

- **Tokenize once, freeze, replay the identical batch sequence** across every
  run — same seed, same order. Data variance must never masquerade as a
  precision effect.
- **Fixed TOKEN budget** across rungs. Not epochs, not wall-clock. The budget is
  a flag (`--steps` / `ONEBIT_STEPS`), never an edited constant, and the resume
  key includes it.
- **3 seeds minimum**, mean and spread both reported. Min–max band, never a
  standard error — with 3 seeds the range *is* the data.
- **Metric: validation loss / perplexity vs tokens.** No downstream benchmarks
  at this scale; they are noise.
- **Every config fully recorded** with each result row (`spec` column).

### The budget is not a free parameter

The headline finding of this project is that **2.05M and 20.48M tokens give
opposite answers** about R5: +0.0% at the small budget, +32.8% at the large one.
A quantization result reported at the wrong budget is not a noisy result, it is
a wrong one, and no amount of seeds or LR sweeping protects against it —
R5-solo survived a full LR sweep at 3 seeds with non-overlapping ranges and was
still an artifact.

---

## Failure classification

Every failed run is classified before it is called a failure:

| class | meaning |
|---|---|
| **(a) divergence** | loss → NaN/inf |
| **(b) stall** | loss flat, `frozen_frac` high — check the freeze law first |
| **(c) collapse** | loss pinned at the random baseline |
| **(d) slower but converging** | **NOT a failure** |

(d) is rerun at 2–5× the token budget before anything is declared broken. Given
what the budget did to R5, this is the protocol that matters most.

---

## Reproduce

```bash
python experiments/m5_ladder.py --steps 10000            # 20.48M tokens, all rungs
python experiments/m5_ladder.py --only R6_attn8,R7_head8 # one or more rungs
python experiments/m4_audit.py                           # the float-op census
python results/plot_m5.py --results results/gpu2         # figures
```

Results append after every run, so the job resumes after a crash.
