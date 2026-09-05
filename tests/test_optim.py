"""M1 tests: the integer optimizer on a convex quadratic.

RECONSTRUCTED after the working tree was deleted.

The M1 gate: if it cannot descend a paraboloid, nothing else matters. Beyond the
gate, these pin the FREEZE LAW -- the one prediction in this project that held
exactly, across 1.8M coordinate-checks and three curvature spectra, and that
went on to predict the M2 and M5 round-to-nearest collapses quantitatively.

Smaller problem (d=200) than the reported experiment (d=1000) so the suite stays
fast; the laws are dimension-independent.
"""

from __future__ import annotations

import pytest
import torch

from experiments.convex_problem import make_quadratic, verify_problem
from optim import IntSGD
from quant import QuantSpec

D = 200
STEPS = 1500
SHADOW_RANGE = 8.0


def problem(kappa=100.0, seed=0, spectrum="logspace"):
    return make_quadratic(d=D, kappa=kappa, seed=seed, spectrum=spectrum)


def spec_for(bits, rounding="nearest"):
    return QuantSpec(kind="int", bits=bits, granularity="tensor", rounding=rounding)


def optimize(p, *, bits=None, mode="rtn", steps=STEPS, seed=0, lr=None):
    lr = p.lr_opt if lr is None else lr
    w = torch.zeros(p.d, requires_grad=True)
    w0 = w.detach().clone()
    if bits is None:
        opt = IntSGD([w], lr=lr, seed=seed)
    else:
        opt = IntSGD([w], lr=lr,
                     shadow_spec=spec_for(bits, "stochastic" if mode == "sr" else "nearest"),
                     shadow_range=SHADOW_RANGE,
                     error_feedback=(mode == "ef"), seed=seed)
    for _ in range(steps):
        w.grad = p.grad(w.detach())
        opt.step()
    return w.detach(), w0, opt


# --------------------------------------------------- the problem is correct

@pytest.mark.parametrize("kappa", [1.0, 10.0, 100.0])
def test_quadratic_matches_its_closed_form(kappa):
    v = verify_problem(problem(kappa))
    assert abs(v["kappa_measured"] - kappa) / kappa < 1e-3
    assert v["grad_at_optimum_rel"] < 1e-5
    assert v["gap_at_optimum"] < 1e-12
    assert v["symmetry_err"] == 0.0


@pytest.mark.parametrize("spectrum", ["logspace", "powerlaw", "bulk_outlier"])
def test_every_spectrum_has_the_declared_condition_number(spectrum):
    """The spectrum experiment only means anything if all three spectra hold
    kappa fixed and vary ONLY the shape."""
    v = verify_problem(problem(100.0, spectrum=spectrum))
    assert abs(v["kappa_measured"] - 100.0) / 100.0 < 1e-3


# ------------------------------------------------------------- THE M1 GATE

def test_gate_fp32_shadow_descends_the_paraboloid():
    """M1 GATE. If this fails, stop and fix before anything else."""
    p = problem(100.0)
    w, w0, _ = optimize(p, bits=None)
    assert p.rel_gap(w, w0) < 1e-3


@pytest.mark.parametrize("mode", ["rtn", "sr", "ef"])
def test_gate_quantized_shadow_descends_the_paraboloid(mode):
    p = problem(100.0)
    w, w0, _ = optimize(p, bits=16, mode=mode)
    assert p.rel_gap(w, w0) < 1e-3, f"{mode} failed the gate"


def test_unquantized_intsgd_matches_plain_gradient_descent_exactly():
    """Cross-check against an independent implementation. The reference arm must
    not differ from textbook GD by even one ulp, or every comparison is
    confounded by the optimizer rather than by precision."""
    p = problem(100.0)
    w_opt, _, _ = optimize(p, bits=None, steps=300)
    w = torch.zeros(p.d)
    for _ in range(300):
        w = w - p.lr_opt * p.grad(w)
    assert torch.equal(w_opt, w)


def test_convergence_rate_matches_theory():
    """rel_gap is quadratic in the error, so it contracts at rho^2 per step
    where rho = (k-1)/(k+1)."""
    p = problem(100.0)
    w = torch.zeros(p.d, requires_grad=True)
    w0 = w.detach().clone()
    opt = IntSGD([w], lr=p.lr_opt)
    gaps = []
    for _ in range(400):
        w.grad = p.grad(w.detach())
        opt.step()
        gaps.append(p.rel_gap(w.detach(), w0))
    measured = (gaps[-1] / gaps[200]) ** (1.0 / (len(gaps) - 201))
    assert abs(measured / p.rate_opt**2 - 1.0) < 0.02


# ------------------------------------------------------------ THE FREEZE LAW

def test_freeze_law_holds_exactly():
    """Under round-to-nearest, coordinate i freezes IFF |lr*g_i| < delta/2.

    The strongest correctness check in the project: it ties the optimizer, the
    quantizer and the problem together with no free parameters. Measured with
    ZERO violations across 1.8M coordinate-checks, and it went on to predict the
    M2 and M5 RTN collapses quantitatively.
    """
    p = problem(100.0)
    lr, bits = p.lr_opt, 8
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=lr, shadow_spec=spec_for(bits),
                 shadow_range=SHADOW_RANGE, seed=0)
    delta = opt.delta
    assert delta == SHADOW_RANGE / (2 ** (bits - 1) - 1)

    violations = checked = 0
    for t in range(STEPS):
        g = p.grad(w.detach())
        w.grad = g
        before = w.detach().clone()
        opt.step()
        if t >= STEPS - 100:
            frozen = w.detach() == before
            mag = (lr * g).abs()
            violations += int((mag[frozen] >= delta / 2).sum())
            violations += int((mag[~frozen] < delta / 2).sum())
            checked += p.d
    assert checked > 0
    assert violations == 0, f"{violations}/{checked} freeze-law violations"


def test_shadow_weights_start_on_the_grid_and_never_saturate():
    p = problem(10.0)
    w = torch.full((p.d,), 0.3141592, requires_grad=True)
    IntSGD([w], lr=0.1, shadow_spec=spec_for(8), shadow_range=SHADOW_RANGE)
    delta = SHADOW_RANGE / 127
    assert (w.detach() / delta - torch.round(w.detach() / delta)).abs().max() < 1e-5

    p2 = problem(100.0)
    w2 = torch.zeros(p2.d, requires_grad=True)
    opt = IntSGD([w2], lr=p2.lr_opt, shadow_spec=spec_for(8), shadow_range=SHADOW_RANGE)
    worst = 0.0
    for _ in range(500):
        w2.grad = p2.grad(w2.detach())
        opt.step()
        worst = max(worst, opt.last["shadow_sat_frac"])
    assert worst == 0.0, "the M1 analysis assumes the latent range is never clipped"


def test_rtn_can_come_to_a_complete_stop():
    """Round-to-nearest genuinely freezes: the parameter stops changing at all.
    The swamping failure mode, reproduced deterministically."""
    p = problem(10.0)
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=p.lr_opt, shadow_spec=spec_for(8), shadow_range=SHADOW_RANGE)
    for _ in range(1500):
        w.grad = p.grad(w.detach())
        opt.step()
    snap = w.detach().clone()
    for _ in range(100):
        w.grad = p.grad(w.detach())
        opt.step()
    assert torch.equal(w.detach(), snap), "expected a complete freeze"
    assert opt.last["frozen_frac"] == 1.0


def test_stochastic_rounding_never_completely_freezes():
    p = problem(10.0)
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=p.lr_opt, shadow_spec=spec_for(8, "stochastic"),
                 shadow_range=SHADOW_RANGE, seed=0)
    for _ in range(1500):
        w.grad = p.grad(w.detach())
        opt.step()
    snap = w.detach().clone()
    moved = False
    for _ in range(100):
        w.grad = p.grad(w.detach())
        opt.step()
        moved |= not torch.equal(w.detach(), snap)
    assert moved, "SR should keep moving forever"


def test_error_feedback_error_scales_with_grid_step():
    """EF carries a residual bounded by delta/2, so its distance from the
    optimum is proportional to delta -- constant when measured in GRID STEPS."""
    p = problem(100.0)
    errs = {}
    for bits in (8, 10, 12):
        w, _, opt = optimize(p, bits=bits, mode="ef")
        errs[bits] = (w - p.w_star).abs().max().item() / opt.delta
    vals = list(errs.values())
    assert max(vals) / min(vals) < 2.0, f"error not proportional to delta: {errs}"
    assert all(v < 6.0 for v in vals)


def test_error_feedback_has_smaller_inf_norm_error_than_rtn():
    """EF tracks far more tightly in the inf-norm. NOTE this does NOT imply a
    better loss -- on a heavy-tailed spectrum EF concentrates its error in the
    stiffest directions and loses to SR on the objective."""
    p = problem(100.0)
    w_rtn, _, o = optimize(p, bits=8, mode="rtn")
    w_ef, _, _ = optimize(p, bits=8, mode="ef")
    assert ((w_ef - p.w_star).abs().max().item()
            < (w_rtn - p.w_star).abs().max().item())


# --------------------------------------------------------------- determinism

@pytest.mark.parametrize("mode", ["rtn", "sr", "ef"])
def test_same_seed_gives_identical_trajectory(mode):
    p = problem(100.0)
    a, _, _ = optimize(p, bits=8, mode=mode, steps=200, seed=7)
    b, _, _ = optimize(p, bits=8, mode=mode, steps=200, seed=7)
    assert torch.equal(a, b)


def test_only_stochastic_mode_depends_on_the_seed():
    p = problem(100.0)
    for mode in ("rtn", "ef"):
        a, _, _ = optimize(p, bits=8, mode=mode, steps=200, seed=0)
        b, _, _ = optimize(p, bits=8, mode=mode, steps=200, seed=1)
        assert torch.equal(a, b), f"{mode} is deterministic; seed must not matter"
    a, _, _ = optimize(p, bits=8, mode="sr", steps=200, seed=0)
    b, _, _ = optimize(p, bits=8, mode="sr", steps=200, seed=1)
    assert not torch.equal(a, b)


# ------------------------------------------------------------- API guardrails

def test_quantized_shadow_requires_an_explicit_fixed_range():
    """A calibrated latent grid would shrink as weights converge and hide the
    stall. The API must refuse to let that happen by accident."""
    w = torch.zeros(10, requires_grad=True)
    with pytest.raises(ValueError, match="shadow_range"):
        IntSGD([w], lr=0.1, shadow_spec=spec_for(8))


def test_shadow_spec_must_be_per_tensor_and_ef_needs_a_shadow():
    w = torch.zeros(64, requires_grad=True)
    with pytest.raises(ValueError, match="granularity"):
        IntSGD([w], lr=0.1,
               shadow_spec=QuantSpec(kind="int", bits=8, granularity="block",
                                     block_size=16),
               shadow_range=1.0)
    with pytest.raises(ValueError, match="error_feedback"):
        IntSGD([torch.zeros(10, requires_grad=True)], lr=0.1, error_feedback=True)


# ---------------------------------- optimizer-side no-op guard (the R5 rung)

def test_shadow_quantizer_noop_guard_passes_when_active():
    """quant.registry covers sites inside the MODEL. Latent-weight quantization
    happens in the optimizer, so R5 -- the crux rung -- needs its own guard."""
    p = problem(100.0)
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=p.lr_opt, shadow_spec=spec_for(8),
                 shadow_range=SHADOW_RANGE, seed=0)
    w.grad = p.grad(w.detach())
    opt.step()
    opt.assert_quantizers_active()
    assert opt.shadow_changed_ever


def test_shadow_guard_fires_when_quantization_is_inert():
    p = problem(100.0)
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=p.lr_opt, shadow_spec=spec_for(8),
                 shadow_range=SHADOW_RANGE, seed=0)
    w.grad = p.grad(w.detach())
    opt.step()
    opt.shadow_changed_ever = False
    with pytest.raises(AssertionError, match="silent no-op"):
        opt.assert_quantizers_active()


def test_shadow_guard_detects_quantization_even_under_total_freeze():
    """A frozen weight has NOT moved, but the quantizer still changed the update
    it was handed. The guard must distinguish 'quantizer inert' from 'weight
    frozen', or a genuine RTN freeze gets misreported as a wiring bug."""
    p = problem(100.0)
    w = torch.zeros(p.d, requires_grad=True)
    opt = IntSGD([w], lr=1e-9, shadow_spec=spec_for(8),
                 shadow_range=SHADOW_RANGE, seed=0)
    before = w.detach().clone()
    w.grad = p.grad(w.detach())
    opt.step()
    assert torch.equal(w.detach(), before), "expected a total freeze"
    assert opt.last["frozen_frac"] == 1.0
    opt.assert_quantizers_active()   # must NOT raise
    assert opt.shadow_changed_ever


def test_gradient_quantization_changes_the_update():
    p = problem(10.0)
    wa = torch.zeros(p.d, requires_grad=True)
    oa = IntSGD([wa], lr=p.lr_opt, seed=0)
    wb = torch.zeros(p.d, requires_grad=True)
    ob = IntSGD([wb], lr=p.lr_opt, seed=0,
                grad_spec=QuantSpec(kind="int", bits=4, granularity="tensor",
                                    rounding="stochastic"))
    for _ in range(50):
        wa.grad = p.grad(wa.detach()); oa.step()
        wb.grad = p.grad(wb.detach()); ob.step()
    assert not torch.equal(wa.detach(), wb.detach())
    assert ob.grad_quant_changed_ever
