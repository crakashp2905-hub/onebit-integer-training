"""M0 correctness tests for the quantization primitives.

RECONSTRUCTED after the working tree was deleted. The original suite (138 tests)
is gone; this restores the checks that carried real weight. Where a test caught
a genuine bug during the project, that is noted, because those are the ones that
justify their existence.

Quantization fails silently, so these are deliberately paranoid. The guiding
rule: every test must be able to FAIL if the quantizer secretly returns its
input, secretly ignores the spec, or secretly rounds the wrong way.
"""

from __future__ import annotations

import math

import pytest
import torch

from quant import (
    ErrorFeedback,
    QuantSpec,
    codes_dtype,
    dequantize,
    fake_quant,
    fake_quant_ste,
    group_view,
    make_generator,
    quantize,
    quantize_backward,
    quantize_ternary,
    registry,
    stochastic_round,
    to_dyadic,
    to_pow2,
)

SPEC_MATRIX = [
    QuantSpec(kind="ternary", granularity="tensor", calib="absmean"),
    QuantSpec(kind="ternary", granularity="row", calib="absmean"),
    QuantSpec(kind="ternary", granularity="tensor", calib="absmean", scale_mode="pow2"),
    QuantSpec(kind="int", bits=8, granularity="tensor"),
    QuantSpec(kind="int", bits=8, granularity="row"),
    QuantSpec(kind="int", bits=8, granularity="col"),
    QuantSpec(kind="int", bits=8, granularity="block", block_size=32),
    QuantSpec(kind="int", bits=4, granularity="row"),
    QuantSpec(kind="int", bits=2, granularity="row"),
    QuantSpec(kind="int", bits=8, granularity="row", scale_mode="pow2", pow2_mode="ceil"),
    QuantSpec(kind="int", bits=8, granularity="row", scale_mode="pow2", pow2_mode="round"),
    QuantSpec(kind="int", bits=8, granularity="row", scale_mode="dyadic", mantissa_bits=3),
    QuantSpec(kind="int", bits=8, granularity="row", rounding="stochastic"),
    QuantSpec(kind="int", bits=16, granularity="row"),
]
IDS = [s.describe() for s in SPEC_MATRIX]


def sample(shape=(8, 64), seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def unique_per_group(t: torch.Tensor, spec: QuantSpec) -> int:
    view, reduce_dim = group_view(t, spec)
    if reduce_dim == 0:
        view = view.t()
    return max(int(torch.unique(row).numel()) for row in view)


# --------------------------------------- 1. stochastic rounding is unbiased

@pytest.mark.parametrize("x", [0.0, 1.0, 4.0, -3.0, 0.5, 0.25, -0.75, 2.3, -2.3, 0.001])
def test_sr_unbiased(x):
    """E[SR(x)] == x, with the EXACT binomial standard error as tolerance.

    SR(x) is Bernoulli on {floor(x), floor(x)+1} with p = frac(x), so the
    standard error of the mean over N draws is sqrt(p(1-p)/N). Using a fudge
    factor instead would let a biased quantizer through.
    """
    n = 200_000
    out = stochastic_round(torch.full((n,), x, dtype=torch.float32), make_generator(1234))
    f = x - math.floor(x)
    sigma = math.sqrt(f * (1.0 - f) / n)
    err = abs(out.mean().item() - x)
    if sigma == 0.0:
        assert err == 0.0, "exact integers must be preserved with probability 1"
    else:
        assert err < 4 * sigma, "bias %.3e exceeds 4 sigma = %.3e" % (err, 4 * sigma)


@pytest.mark.parametrize("x", [0.3, -0.3, 2.7, -2.7, 5.0])
def test_sr_support_is_floor_or_ceil(x):
    out = stochastic_round(torch.full((10_000,), x), make_generator(7))
    assert set(torch.unique(out).tolist()) <= {math.floor(x), math.ceil(x)}


# ------------------------------------------------ 2. ternary is exactly ternary

def test_ternary_codes_are_exactly_ternary():
    r = quantize_ternary(sample((128, 128)))
    assert not r.codes.is_floating_point()
    levels = set(torch.unique(r.codes).tolist())
    assert levels <= {-1, 0, 1}
    assert levels == {-1, 0, 1}, "degenerate: not all three levels used"


def test_ternary_beta_is_absmean_and_matches_theory():
    """Cross-check against closed form: for Gaussian W with beta = E|W|, a weight
    rounds to zero when |W| < beta/2, i.e. |z| < 0.399, probability ~0.310."""
    w = sample((256, 256))
    r = quantize_ternary(w)
    assert math.isclose(r.scale.item(), w.abs().mean().item(), rel_tol=1e-6)
    assert abs(r.stats["zero_frac"] - 0.310) < 0.02


# ------------------------------------------------------------ 3. STE behaviour

def test_ste_identity_passes_gradient_while_forward_quantizes():
    x = sample().requires_grad_(True)
    y = fake_quant_ste(x, spec=QuantSpec(kind="int", bits=4, granularity="row"))
    assert not torch.equal(y.detach(), x.detach()), "forward is not quantized"
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_ste_clipped_masks_outside_range():
    x = sample().requires_grad_(True)
    spec = QuantSpec(kind="int", bits=4, granularity="row", ste="clipped", ste_clip=0.5)
    fake_quant_ste(x, spec=spec).sum().backward()
    expected = (x.detach().abs() <= 0.5).to(x.dtype)
    assert torch.equal(x.grad, expected)
    assert 0.0 < expected.mean().item() < 1.0


def test_ste_none_gives_true_zero_gradient():
    x = sample().requires_grad_(True)
    fake_quant_ste(x, spec=QuantSpec(kind="int", bits=4, granularity="row",
                                     ste="none")).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


def test_ste_has_no_second_gradient_path():
    """Quantization must run under no_grad; otherwise gradient also flows through
    the scale arithmetic and would not equal exactly 1."""
    x = sample((4, 32)).requires_grad_(True)
    fake_quant_ste(x, spec=QuantSpec(kind="int", bits=8, granularity="tensor")).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


# ------------------------------------------------------ 4. NO-OP DETECTOR

@pytest.mark.parametrize("spec", SPEC_MATRIX, ids=IDS)
def test_quantizer_is_not_a_noop(spec):
    x = sample()
    q = fake_quant(x, spec=spec, generator=make_generator(5))
    assert not torch.equal(q, x), "quantizer returned its input unchanged"
    assert (q != x).float().mean().item() > 0.5


@pytest.mark.parametrize("spec", SPEC_MATRIX, ids=IDS)
def test_quantizer_respects_level_budget(spec):
    """A group may not hold more distinct values than the grid has levels. A
    pass-through would show ~numel distinct values."""
    q = fake_quant(sample(), spec=spec, generator=make_generator(5))
    assert unique_per_group(q, spec) <= spec.n_levels


def test_kind_none_is_a_passthrough_by_design():
    x = sample((4, 16))
    assert torch.equal(fake_quant(x, spec=QuantSpec(kind="none")), x)


# ---------------------------------------- 5. quantize/dequantize agree exactly

@pytest.mark.parametrize("spec", SPEC_MATRIX, ids=IDS)
def test_dequantize_of_codes_equals_fake_quant(spec):
    """The Track A / Track B honesty invariant: the float path the model sees
    must be EXACTLY what you get from the integer codes and the scale."""
    x = sample()
    r = quantize(x, spec=spec, generator=make_generator(5))
    assert torch.equal(dequantize(r.codes, r.scale, spec), r.q)
    assert int(r.codes.abs().max().item()) <= spec.qmax


# ------------------------------------------------------ 6. scale representation

def test_pow2_scales_are_exactly_powers_of_two():
    for mode in ("ceil", "round"):
        spec = QuantSpec(kind="int", bits=8, granularity="row",
                         scale_mode="pow2", pow2_mode=mode)
        m, _ = torch.frexp(quantize(sample((16, 64)), spec=spec).scale)
        assert torch.all(m == 0.5), f"{mode}: scales are not powers of two"


def test_pow2_ceil_never_saturates_and_is_idempotent():
    spec = QuantSpec(kind="int", bits=8, granularity="row",
                     scale_mode="pow2", pow2_mode="ceil")
    assert quantize(sample((16, 64)), spec=spec).stats["sat_frac"] == 0.0
    s = torch.tensor([8.0, 0.3, 1.0, 1e-8, 12345.0])
    once = to_pow2(s, "ceil")
    assert torch.equal(to_pow2(once, "ceil"), once)
    assert torch.all(once >= s) and torch.all(once < 2 * s)


def test_dyadic_converges_to_the_float_scale():
    s = torch.rand(1000) * 100 + 1e-6
    errs = [(to_dyadic(s, k) - s).abs().div(s).max().item() for k in (0, 2, 4, 8, 16)]
    assert errs == sorted(errs, reverse=True), "more mantissa bits must not hurt"
    assert errs[-1] < 1e-4


def test_scale_mode_actually_changes_the_result():
    x = sample()
    base = fake_quant(x, spec=QuantSpec(kind="int", bits=8, granularity="row"))
    p2 = fake_quant(x, spec=QuantSpec(kind="int", bits=8, granularity="row",
                                      scale_mode="pow2"))
    assert not torch.equal(base, p2), "scale_mode='pow2' had no effect"


# ------------------------------------------------------------ 7. honest stats

def test_stats_match_independent_computation():
    spec = QuantSpec(kind="int", bits=4, granularity="row", rounding="nearest")
    x = sample() * 3.0
    r = quantize(x, spec=spec)
    view, _ = group_view(x, spec)
    rounded = torch.round(view / r.scale)
    sat = (rounded.abs() > spec.qmax).float().mean().item()
    zero = (rounded.clamp(-spec.qmax, spec.qmax) == 0).float().mean().item()
    assert math.isclose(r.stats["sat_frac"], sat, abs_tol=1e-9)
    assert math.isclose(r.stats["zero_frac"], zero, abs_tol=1e-9)


def test_underflow_is_reported_not_hidden():
    r = quantize(torch.full((64,), 1e-9),
                 spec=QuantSpec(kind="int", bits=8, granularity="tensor"),
                 fixed_scale=torch.tensor(1.0))
    assert r.stats["zero_frac"] == 1.0
    assert torch.equal(r.q, torch.zeros(64))


# -------------------------------------------------------------- 8. RNG hygiene

def test_same_seed_identical_different_seed_differs():
    x = sample((512,)) * 2.0
    assert torch.equal(stochastic_round(x, make_generator(42)),
                       stochastic_round(x, make_generator(42)))
    assert not torch.equal(stochastic_round(x, make_generator(42)),
                           stochastic_round(x, make_generator(43)))


def test_stochastic_rounding_does_not_disturb_the_global_rng():
    """If SR consumed the global stream, batch order would change whenever a
    quantizer changed, confounding precision effects with data."""
    x = sample((4096,))
    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)
    stochastic_round(x, make_generator(99))
    assert torch.equal(before, torch.rand(4))


# ------------------------------- 9. error feedback vs swamping (the R5 crux)

def _swamp(step=1.0, value=0.05, n=1):
    return (QuantSpec(kind="int", bits=8, granularity="tensor"),
            torch.full((n,), value), torch.tensor(step))


def test_round_to_nearest_annihilates_sub_step_values():
    """The failure this project predicted for quantized shadow weights."""
    spec, x, step = _swamp()
    for _ in range(200):
        assert torch.equal(quantize(x, spec=spec, fixed_scale=step).q, torch.zeros_like(x))


def test_error_feedback_recovers_the_mean_and_bounds_its_residual():
    spec, x, step = _swamp()
    ef = ErrorFeedback(spec, fixed_scale=step)
    total, worst, t = torch.zeros_like(x), 0.0, 2000
    for _ in range(t):
        total += ef.apply(x)
        worst = max(worst, ef.last_stats["residual_absmax"])
    assert abs((total / t).item() - 0.05) < 0.002
    assert worst <= 0.5 + 1e-6, "residual must stay bounded by step/2"


# --------------------------------------------------------------- 10. contracts

@pytest.mark.parametrize("spec", SPEC_MATRIX, ids=IDS)
def test_shape_dtype_and_zeros_are_safe(spec):
    x = sample()
    r = quantize(x, spec=spec, generator=make_generator(1))
    assert r.q.shape == x.shape and r.codes.shape == x.shape
    assert r.q.dtype == x.dtype and not r.codes.is_floating_point()
    assert torch.isfinite(r.q).all()
    z = quantize(torch.zeros(8, 64), spec=spec, generator=make_generator(1))
    assert torch.isfinite(z.q).all() and torch.equal(z.q, torch.zeros(8, 64))


def test_codes_dtype_is_narrowest_that_fits():
    assert codes_dtype(QuantSpec(kind="ternary")) is torch.int8
    assert codes_dtype(QuantSpec(kind="int", bits=8)) is torch.int8
    assert codes_dtype(QuantSpec(kind="int", bits=16)) is torch.int16
    assert codes_dtype(QuantSpec(kind="int", bits=24)) is torch.int32


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        QuantSpec(kind="int", bits=1)
    with pytest.raises(ValueError):
        QuantSpec(granularity="block")
    with pytest.raises(TypeError):
        quantize(torch.arange(16), spec=QuantSpec(kind="int", bits=8))
    with pytest.raises(ValueError):
        quantize(sample(), spec=QuantSpec(kind="int", bits=8,
                                          granularity="block", block_size=7))


def test_floor_rounding_is_biased_as_a_control():
    x = sample((4096,)) * 0.5
    q = fake_quant(x, spec=QuantSpec(kind="int", bits=4, granularity="tensor",
                                     rounding="floor"))
    assert q.mean().item() < x.mean().item()


# ------------------------------------------------ 11. gradient-path quantizer

def test_quantize_backward_is_identity_forward_and_quantizes_the_gradient():
    x = sample((4, 32)).requires_grad_(True)
    spec = QuantSpec(kind="int", bits=4, granularity="row")
    y = quantize_backward(x, spec=spec, generator=make_generator(0))
    assert torch.equal(y.detach(), x.detach()), "forward must be identity"
    g = torch.randn(4, 32)
    y.backward(g)
    assert not torch.equal(x.grad, g), "gradient was not quantized"
    assert max(torch.unique(r).numel() for r in x.grad) <= spec.n_levels


# ------------------------------------------------------------ 12. registry

def test_registry_catches_a_silent_noop_and_a_never_called_site():
    registry.reset()
    x = sample((4, 16))
    registry.record("fake.site", x, quantize(x, spec=QuantSpec(kind="none")))
    registry.sites()["fake.site"].spec = QuantSpec(kind="int", bits=8)
    with pytest.raises(AssertionError, match="silent no-op"):
        registry.assert_no_noops()
    registry.reset()
    registry.register_site("never.called", QuantSpec(kind="int", bits=8))
    with pytest.raises(AssertionError, match="never called"):
        registry.assert_no_noops()
    registry.reset()


def test_registry_passes_for_a_real_quantizer():
    registry.reset()
    fake_quant_ste(sample((4, 16)),
                   spec=QuantSpec(kind="int", bits=4, granularity="row"),
                   site="real.site")
    registry.assert_no_noops()
    rows = registry.dump_stats()
    assert len(rows) == 1 and rows[0]["changed_ever"] is True
    registry.reset()


def test_pow2_ceil_changes_ternary_sparsity_not_just_the_scale():
    """Pins the confound behind every pow2 ladder result. Rounding the ternary
    scale UP to a power of two raises the zero threshold, so 'pow2 weights'
    also means 'sparser weights' -- measured 31% -> 56% zeros at std 0.02. If
    this ever stops holding (e.g. pow2_mode changes), the §2 correction in
    RESULTS.md needs revisiting."""
    torch.manual_seed(0)
    W = torch.randn(576, 192) * 0.02
    f = quantize(W, spec=QuantSpec(kind="ternary", granularity="tensor", calib="absmean"))
    p = quantize(W, spec=QuantSpec(kind="ternary", granularity="tensor", calib="absmean",
                                   scale_mode="pow2", pow2_mode="ceil"))
    zf = (f.codes == 0).float().mean().item()
    zp = (p.codes == 0).float().mean().item()
    assert p.scale.item() > f.scale.item()
    assert zp > zf + 0.10, f"ceil-pow2 should materially sparsify ternary: {zf:.3f} vs {zp:.3f}"


def test_scale_mult_scales_the_scale_and_rejects_nonpositive():
    x = torch.randn(64, 32)
    a = quantize(x, spec=QuantSpec(kind="ternary", calib="absmean"))
    b = quantize(x, spec=QuantSpec(kind="ternary", calib="absmean", scale_mult=1.5))
    assert torch.allclose(b.scale, a.scale * 1.5)
    with pytest.raises(ValueError):
        QuantSpec(kind="ternary", scale_mult=0.0)


def test_pow2_round_control_leaves_ternary_sparsity_roughly_unchanged():
    """C3 is only a clean representation-only control if round-to-nearest pow2
    does NOT reproduce ceil's sparsification. Measured 30.3% vs float 31.0%."""
    torch.manual_seed(0)
    W = torch.randn(576, 192) * 0.02
    zf = (quantize(W, spec=QuantSpec(kind="ternary", calib="absmean")).codes == 0).float().mean()
    zr = (quantize(W, spec=QuantSpec(kind="ternary", calib="absmean", scale_mode="pow2",
                                     pow2_mode="round")).codes == 0).float().mean()
    assert abs(zr - zf) < 0.05
