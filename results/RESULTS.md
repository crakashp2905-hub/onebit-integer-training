# Results

**Track A: how far can a transformer training loop be pushed to integer/low
precision before it breaks, and why.**

Every number here came from a run that actually executed. Reproduce commands are
given with each section.

> **Provenance note.** The working tree was permanently deleted twice while in
> `Downloads` (a cleanup process; free space jumped 12 GB → 47 GB, nothing in the
> Recycle Bin). The first loss destroyed the original CSVs, plots, and 12 commits
> of history; the code survived only because it had been pushed to Kaggle for a
> GPU run. The project now lives outside `Downloads` with a GitHub remote.
> Sections marked **[restored]** were rebuilt from the conversation record and
> re-verified where re-running was cheap. M0 reproduced bit-identically on
> re-run, which is the main evidence that the recovered code is faithful.

---

## The headline

At a realistic token budget, **quantizing the latent weight is expensive, and
the small-budget result that said otherwise was an artifact of undertraining.**

| config | ppl @20.5M tok | vs fp32 @20.5M | vs fp32 @2.05M (CPU) |
|---|---:|---:|---:|
| R0_fp32 | 7.638 | — | — |
| R1_ternary | 9.841 | **+28.8%** | +8.6% |
| R2_act8 | 9.846 | +28.9% | +7.3% |
| R2p5_pow2scales | 9.259 | **+21.2%** | +6.7% |
| R5_shadow8ef | 10.143 | **+32.8%** | **+0.0%** |
| R5_shadow8sr | 10.210 | +33.7% | +0.3% |

3 seeds except R2p5 and R5_shadow8sr (n=1). Model: 3.09M-param decoder,
d=192 L=6 H=6 ctx=128 vocab=2048, 86.5% of parameters in the transformer body.

Three things follow, and they are the substance of the project.

### 1. The undertrained regime masks quantization damage

The FP32 baseline improves enormously with 10× tokens (ppl 22.29 → 7.638) while
ternary improves far less (24.22 → 9.841). The *relative* cost of ternary
weights therefore **triples**, +8.6% → +28.8%.

Every margin measured at the small budget is understated. This is not a minor
caveat; it is the dominant effect in the whole dataset.

### 2. A 3-seed, lr-swept, non-overlapping result was still an artifact

At 2.05M tokens, 8-bit latent weights matched FP32 exactly (+0.0%), and the solo
variant appeared to **beat** it by 3.7%. That did not look like noise, and it
was not: a full learning-rate sweep showed the untuned lr=0.5 explained only
part of it. At each arm's own optimum, over 3 seeds:

| arm | best lr | ppl | spread |
|---|---:|---:|---|
| fp32 | 0.25 | 21.762 | [21.677–21.826] |
| shadow8ef | 0.35 | **21.258** | [21.133–21.463] |

A 2.3% advantage, ranges non-overlapping, ~3.3× the larger standard deviation.

**At 20.5M tokens it is +32.8% worse, ranges non-overlapping.**

Quantizing the latent weight acts as a regularizer. It helps only while the
model is ~30× short of Chinchilla-optimal and has nothing to overfit away from.
Given a realistic budget the regularizer becomes pure damage.

The methodological lesson is sharper than the result: **effect size, seed
spread, and hyperparameter sweeps do not protect against a systematically wrong
regime.** Everything that could be checked *within* the small budget was
checked, and the conclusion still inverted.

### 3. Multiplier-free scaling is close to free; bit-width is what costs

R2p5 constrains **every** scale to a power of two, so requantization is a shift
rather than a float multiply. At 20.5M tokens it costs +21.2% against R2's
+28.9% — i.e. no worse, and possibly better (n=1, so read the ordering, not the
margin).

This is the fourth independent line of evidence against the kickoff prediction
that scale representation would bind before bit-width, after M0 static error,
M0 granularity, and the CPU ladder. Scope: 8 bits. M0 measured the pow2 penalty
growing to +68% at 3 bits, and no 4-bit pow2 rung has been run.

---

## M0 — quantizer primitives

```
python -m pytest                          ->  140 passed  [restored]
python experiments/m0_characterize.py     ->  results/m0_quantizer_error.csv
```

Re-run after the data loss and reproduced **bit-identically**.

### Is power-of-two scaling more expensive than one bit of width?

NRMSE, Gaussian, per-row scaling. Lower is better.

| bits | float scale | pow2/ceil | pow2/round | float @ (bits−1) |
|---:|---:|---:|---:|---:|
| 8 | 0.00788 | 0.00989 (+25%) | 0.00938 (+19%) | 0.01593 (+102%) |
| 6 | 0.03231 | 0.04033 (+25%) | 0.03626 (+12%) | 0.06685 (+107%) |
| 4 | 0.14307 | 0.21537 (+51%) | 0.14434 (+1%) | 0.33406 (+134%) |
| 3 | 0.33406 | 0.56205 (+68%) | 0.29994 (−10%) | 0.86935 (+160%) |

Dropping a bit roughly doubles error; a power-of-two scale costs 12–25% at 6–8
bits. Three qualifications:

1. **The pow2 penalty grows as bits shrink** (+25% at 8 → +68% at 3). The axes
   are not independent: ceil wastes up to an octave of range, a fixed fraction
   of a bit, which matters more when you have three.
2. **pow2/round beating a float scale at 3 bits is a confound.** It picks a
   scale 0.89× the float scale, buying resolution while clipping 0.046% of
   values — implicit outlier clipping, which absmax badly needs at low
   bit-width. A fact about absmax, not about powers of two.
3. **pow2/ceil silently zeroes things.** `zero_frac` at 3 bits: 0.435 → 0.666.
   For weights that is lossy; for gradients it is silent layer death with no NaN.

### Granularity vs outliers (int8)

| granularity | gaussian | 8 outlier channels (30×) | ratio |
|---|---:|---:|---:|
| tensor | 0.01059 | 0.11378 | 10.7× |
| row | 0.00788 | 0.04498 | 5.7× |
| block-64 | 0.00594 | 0.01651 | 2.8× |

Finer granularity is the fix for outliers. Outliers here are synthetic — read
the ordering, not the magnitudes.

---

## M1 — the freeze law  [restored]

Convex quadratic `f(w) = ½wᵀAw − bᵀw`, d=1000, optimum known by construction.

**The one prediction that held exactly.** Under round-to-nearest, a coordinate
already on the grid satisfies

```
Q(wᵢ − lr·gᵢ) = wᵢ   ⟺   |lr·gᵢ| < Δ/2
```

**Zero violations across 1.8M coordinate-checks and three curvature spectra.**
At κ=100, 8 bits the separation is razor-sharp: frozen coordinates top out at
|lr·g| = 0.031086, moving ones bottom out at 0.031712, with Δ/2 = 0.031496
between them.

It then predicted the MNIST collapse *quantitatively before that run happened*:
Δ = 0.00394, freezing requires |g| < 0.0197, actual gradients ~1e−3 — 25× too
small to ever move a weight. Measured: `frozen_frac` 0.999, test loss 2.2984
against ln(10) = 2.3026. The model never left initialization.

### What did NOT generalize

- **The aggregate stall bound was wrong.** Predicted stall at ‖g‖∞ < Δ/(2·lr);
  measured ‖g‖∞ sitting at **4.3×** that bound. The law is per-coordinate: the
  vector partitions into frozen low-curvature coordinates and high-curvature
  ones that limit-cycle at ±Δ forever. In a loss curve that reads as a noisy
  plateau, not a stop.
- **The convex instrument cannot rank non-freezing schemes.** "RTN wins iff it
  fully freezes" held 12/12 on a log-spaced spectrum and collapsed to 6/12
  across spectra; on the NN-like `bulk_outlier` spectrum RTN loses to SR by
  64–175×.
- **Error feedback's guarantee is a per-coordinate ∞-norm bound and says
  nothing about where the error lands.** On `bulk_outlier`, EF drives 77.8% of
  its error energy into 13 of 1000 directions — the stiffest, which the A-norm
  charges ~100× for. Smallest ∞-norm error (14× tighter than RTN), 4× worse loss
  than SR.

### Resolution floor

`torch.linalg.qr` is thread-count dependent, giving a free 7.68e−7 roundoff
perturbation. Amplification into final loss: median 2.7% (RTN), 10.2% (SR), 4.5%
(EF), worst cases 21–34%. **Any effect below that floor is not resolved by seed
spread**, because it is sensitivity to roundoff, not to seeds. Thread count is
now pinned and recorded in every row.

---

## M2 — MNIST MLP  [restored]

MLP 784-256-256-256-10, 1500 steps × batch 128, 3 seeds.

| component | 8-bit | 4-bit |
|---|---:|---:|
| activations | −0.0004 | −0.0010 |
| dgrad | −0.0005 | −0.0010 |
| wgrad | −0.0006 | −0.0004 |
| ternary weights | −0.0074 | — |
| **shadow RTN** | **−0.8324** | **−0.8560** |
| shadow SR | −0.0032 | −0.0718 |
| shadow EF | −0.0004 | −0.0057 |

**Gradients are not the hard rung.** Even 4-bit dgrad and wgrad cost nothing;
ternary weights cost 10× more. This contradicted the project's original
expectation that R3 would break first, on both MNIST and the transformer.

4-bit wgrad reaches `frozen_frac` 0.706 — 71% of weights immobile on a given
step — with **zero** accuracy cost. Sparse updates are fine.

A design bug worth recording: the first MLP was `(784,256,128,10)`, and with the
standard convention of leaving first/last layers in full precision that puts
**86% of parameters outside the ladder**. The full stack scored 0.9662 vs 0.9703
fp32 — inside noise. An experiment that cannot break cannot inform.

---

## M3/M5 — the transformer

```
python experiments/m3_baseline.py   ->  results/m3_baseline.csv
python experiments/m5_ladder.py     ->  results/m5_ladder.csv
kaggle/                             ->  the GPU runs
```

FP32 reference at 2.05M tokens: AdamW ppl 20.23, IntSGD+momentum ppl 22.29,
random ln(2048) = 7.62 nats. Two arms deliberately — every quantized rung is
compared against **IntSGD**, so damage is never confounded with "SGD is worse
than Adam".

**RTN latent weights collapse, and the momentum prediction was half right.**
Recorded before the run: M2 used momentum=0 and RTN fell to chance; M5 uses 0.9,
and momentum is itself an accumulator, so RTN might survive. Measured: ppl 51.96
against a random baseline of 2048 — badly degraded but genuinely learning.
**Momentum is a partial fix for swamping, not a cure.**

### GPU throughput

Kaggle allocated a P100 despite an explicit T4 request. Measured 6,214 tok/s
against 3,577 on 4 CPU cores — only **1.74×**. At batch 16 × ctx 128 = 2,048
tokens/step the device is nowhere near saturated: the bottleneck is
kernel-launch overhead across many tiny elementwise fake-quant ops, not
arithmetic. Any future GPU work needs a much larger batch.

---

## M4 — the float-op audit

```
python experiments/m4_audit.py      ->  results/m4_audit.csv
```

This was in the original build order between M3 and M5, and got skipped. Without
it the project could report what quantization **costs** (every loss number
above) but not what it **bought** — and "multiplier-free" is a claim about
operations, not about loss.

The counter is a `TorchDispatchMode` that attributes every op to a module across
both the forward and the backward pass. It reports two quantities:

- **`fp_mul`** — the FP multiplies a *real* integer kernel would still perform.
  A ternary matmul contributes 0 (multiplying by {−1, 0, +1} is a
  select/negate/skip); its requantization scale costs one FP multiply per output
  element **unless** that scale is a power of two. This is the number that turns
  "we constrained the scales to powers of two" into a measurement.
- **`raw_mul`** — what Track A literally executes. Simulated quantization runs on
  FP units, so this barely moves. That is exactly why `fp_mul` has to be
  modelled rather than measured, and it is the honest statement of what a
  fake-quant study can and cannot claim.

One training step, batch 4 × ctx 128:

| config | fp_mul | vs fp32 | ppl cost @20.5M |
|---|---:|---:|---:|
| R0_fp32 | 5,138,546,688 | 100.0% | — |
| R1_ternary | 3,790,209,048 | 73.8% | +28.8% |
| R2_act8 | 3,798,478,872 | 73.9% | +28.9% |
| R2p5_pow2scales | 3,793,170,456 | 73.8% | +21.2% |
| R3b_wgrad8 | 1,097,981,976 | **21.4%** | **~0%** |
| R5_shadow8ef | 1,097,981,976 | 21.4% | +32.8% |
| R6_attn8 | 958,267,416 | 18.6% | *running* |
| R7_head8 | 758,975,000 | 14.8% | *running* |
| R7_everything_pow2 | 749,668,888 | **14.6%** | *running* |

### The ladder is upside down

**Ternary weights cost +28.8% perplexity and remove only 26% of the multiplies.
Gradient quantization costs nothing measurable and removes another 53 points.**

The rung this project expected to break first is the one carrying most of the
benefit; the rung everyone starts from is the expensive one. This is only
visible because the audit exists — every loss-only view of the ladder gets the
cost/benefit ordering backwards.

The reason is structural, not a quirk of this model. R1 quantizes the *weight*,
so it can only ever claim the one forward matmul. R3 quantizes the *gradient*,
which is an operand of **both** backward matmuls — and the backward pass is two
thirds of the arithmetic in a training step.

### Where the residue lives

Of the 1,097,981,976 multiplies that survived the full stack:

| source | share | in the ladder? |
|---|---:|---|
| LM head (forward 201M + backward 403M) | **55%** | no — it was an `nn.Linear` |
| attention QK^T and AV, all 6 layers | **42%** | no — R6, unbuilt |
| everything else | 3% | — |

97% of the remaining multiplies sat in two places no rung had ever touched, and
the LM head's share was an accident of code structure: it was an `nn.Linear`
buried in `QuantGPT.forward` rather than a `QuantLinear`. It has since been
promoted to its own leaf module (R7) and attention quantization added (R6),
taking the audited residue to 14.6%.

### The irreducible part

**6,815,744 transcendentals per step = 13,312 per token**, identical across
every single rung — softmax `exp`/divide, LayerNorm `rsqrt`, GELU `erf`. No
scaling scheme touches these. Only changing the architecture does. That is the
floor the brief asked this project to locate, and it is the honest answer to
"can a transformer be made multiplier-free": not this transformer.

### A bug worth recording

The first version set the quantized-operand flag in a forward hook only, so
every backward matmul scored as full floating point and the audit reported 26%
elimination for configs whose backward is largely integer. A second version kept
that flag as a scalar — but attention contains `QuantLinear` children whose
`pop` cleared it, and QK^T and AV run *after* `self.qkv` returns, so R6 would
have measured as worth exactly zero. The flag is now a stack parallel to the
label stack, and backward matmuls are scored conservatively: the integer flag is
set only when *both* dgrad and wgrad qualify, so the audit can understate a
rung's benefit but never overstate it.

---

## Predictions made, and how they turned out

| prediction | outcome |
|---|---|
| R5 (latent weights) is the crux rung | **held** — the only rung that breaks, on every instrument |
| Gradients more forgiving than expected | **held** — 4-bit dgrad/wgrad essentially free |
| Freeze law: coord frozen ⟺ \|lr·g\| < Δ/2 | **held exactly** — 0 violations, predicted M2 and M5 |
| Scale representation binds before bit-width | **wrong**, four times over |
| Aggregate stall bound ‖g‖∞ < Δ/(2·lr) | **wrong** — observed 4.3× the bound |
| Error feedback beats stochastic rounding | **flipped three times**; dissolves at scale |
| Momentum may rescue RTN latent weights | **half right** — degraded but learning |
| Parallelism would speed the sweep | **wrong** — memory-bandwidth-bound, 0.97× |
| (unstated, and wrong) that the ladder's rungs cost roughly in proportion to what they buy | **wrong** — ternary buys 26% for +28.8%, gradients buy 53% for free |

Three held, five wrong, one half. The failures cluster: every one was a case of
generalizing from an instrument that could not support the generalization —
a single spectrum, a single token budget, a single problem class.

---

## Open

- **4-bit power-of-two scaling, R6 (attention), R7 (LM head).** Built and
  audited; loss numbers at 20.5M tokens are **running now** and are the one gap
  between the audit's cost column and its benefit column.
- **Attention backward.** R6 quantizes the forward QK^T and AV only, so the
  audit scores dQ/dK/dV as full FP. Roughly two thirds of attention's arithmetic
  is therefore still unclaimed.
- **The transcendentals.** 13,312 per token, untouched by every rung. Reaching
  them means replacing softmax and LayerNorm, not rescaling them.
- **R4 (quantized optimizer state).** IntSGD carries FP32 momentum; only the
  latent weight and weight gradient are quantized.
- **The full ladder at 20.5M tokens.** Only 6 of 12 configs have GPU numbers;
  each Kaggle session caps at 12h and the current batch size wastes most of the
  GPU.
