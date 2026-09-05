"""SGD whose shadow (latent) weights live on a quantized grid.

This is the R5 mechanism in isolation. Standard BitNet keeps the latent weight in
FP32 precisely so that many small updates can accumulate until they cross a
ternarization threshold. Quantizing the latent weight itself is what this class
does, and the predicted failure is a STALL, not a divergence:

    if the grid step is D and the update is lr*g, then round-to-nearest maps
    w - lr*g back to w whenever |lr*g| < D/2, and that coordinate FREEZES.

    Freeze law (exact, per coordinate):   coord i frozen  <=>  |lr * g_i| < D/2

MEASURED at M1 and verified with zero violations. Note what this does NOT say:
it is a per-coordinate law, not a global one. Training does not come to a halt.
The parameter vector partitions into

    - low-curvature coordinates, which fall below the threshold and freeze, and
    - high-curvature coordinates, which stay above it and LIMIT-CYCLE, bouncing
      +/- D forever without ever settling.

The naive aggregate prediction "the run stalls when ||g||_inf < D/(2*lr)" is
therefore FALSE, and was measured to be false at M1 (observed ||g||_inf sat at
4.3x that bound). This matters for R5: quantized shadow weights are expected to
produce a partial freeze plus a persistent limit cycle, not a clean stop, which
looks like a noisy plateau in a loss curve rather than an obvious failure.

M1 uses a convex quadratic precisely so this can be checked rather than assumed:
there the gradient is known exactly at every point.

Three ways to fight the stall, all supported here:

    rounding="nearest",    error_feedback=False   -> RTN. Hard stall. The control.
    rounding="stochastic", error_feedback=False   -> SR.  Unbiased per step, so it
                                                    keeps moving, but injects
                                                    variance: expect a noise ball.
    rounding="nearest",    error_feedback=True    -> EF.  Residual carried forward.
                                                    Bounded error, no injected noise.

IMPORTANT: the latent grid is FIXED (shadow_range / qmax), never calibrated from
the current weights. A dynamically calibrated grid would shrink as the weights
approach the optimum, which would hide the very stall we are trying to measure.
"""

from __future__ import annotations

import torch

from quant import ErrorFeedback, QuantSpec, fake_quant, make_generator, quantize


class IntSGD(torch.optim.Optimizer):
    """SGD with an optionally quantized shadow weight and optionally quantized
    gradient.

    Args:
        params: iterable of tensors (a standard torch.optim params argument).
        lr: learning rate.
        momentum: heavy-ball momentum on the (possibly quantized) gradient.
        shadow_spec: QuantSpec for the latent weight. kind="none" gives exact
            FP32 SGD, which is the reference arm and shares this code path so
            the comparison is not confounded by a different implementation.
        shadow_range: maximum representable latent magnitude. The grid step is
            D = shadow_range / qmax. Required unless shadow_spec.kind == "none".
        grad_spec: QuantSpec for the gradient, applied before the update.
            Calibrated dynamically (absmax) since gradient magnitude changes by
            orders of magnitude over training -- the integer analog of loss
            scaling. kind="none" leaves gradients in FP32.
        error_feedback: carry the latent rounding residual forward.
        seed: base seed. Each parameter gets its own generator derived from
            (seed, param index) so stochastic rounding never touches the global
            RNG stream and runs stay reproducible.
    """

    def __init__(
        self,
        params,
        lr: float,
        *,
        momentum: float = 0.0,
        shadow_spec: QuantSpec | None = None,
        shadow_range: float | str | None = None,
        shadow_range_mult: float = 8.0,
        grad_spec: QuantSpec | None = None,
        error_feedback: bool = False,
        seed: int = 0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        shadow_spec = shadow_spec or QuantSpec(kind="none")
        grad_spec = grad_spec or QuantSpec(kind="none")
        auto_range = shadow_range == "auto"
        if shadow_spec.kind != "none":
            if not auto_range and (shadow_range is None or shadow_range <= 0):
                raise ValueError(
                    "a quantized shadow weight needs an explicit positive "
                    "shadow_range, or 'auto'; the latent grid must be FIXED, "
                    "not calibrated"
                )
            if shadow_spec.granularity != "tensor":
                raise ValueError(
                    "shadow_spec must use granularity='tensor': a fixed latent "
                    "grid is one step for the whole tensor by definition"
                )
        if error_feedback and shadow_spec.kind == "none":
            raise ValueError("error_feedback is meaningless with an unquantized shadow")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

        self.shadow_spec = shadow_spec
        self.grad_spec = grad_spec
        self.shadow_range = shadow_range
        self.shadow_range_mult = shadow_range_mult
        self.error_feedback = error_feedback
        self.seed = seed
        self.auto_range = auto_range

        # delta is a single scalar only when the range is global; with auto
        # ranging each parameter gets its own fixed grid, stored in its state.
        self.delta: float | None = (
            None
            if shadow_spec.kind == "none" or auto_range
            else shadow_range / shadow_spec.qmax
        )

        # per-step diagnostics, overwritten each step()
        self.last: dict[str, float] = {}

        # NO-OP GUARD for the shadow rung. quant.registry covers sites inside
        # the model, but latent-weight quantization happens HERE, so R5 -- the
        # crux rung -- had no silent-no-op detection at all. Tracked the same
        # way: short-circuits permanently once True.
        self.shadow_changed_ever = False
        self.grad_quant_changed_ever = False

        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["gen"] = make_generator(seed * 100_003 + idx, p.device)
                st["grad_gen"] = make_generator(seed * 100_003 + 50_021 + idx, p.device)
                if shadow_spec.kind != "none":
                    if auto_range:
                        # fixed grid sized to the INITIAL weight scale, computed
                        # once and never recalibrated -- a grid that tracked the
                        # weights would hide the swamping we are measuring
                        rng = shadow_range_mult * p.detach().abs().max().item()
                        rng = max(rng, 1e-8)
                    else:
                        rng = float(shadow_range)
                    d = rng / shadow_spec.qmax
                    st["delta"] = d
                    st["shadow_range"] = rng
                    st["fixed_scale"] = torch.tensor(d, dtype=p.dtype, device=p.device)
                    if error_feedback:
                        st["ef"] = ErrorFeedback(
                            shadow_spec,
                            generator=st["gen"],
                            fixed_scale=st["fixed_scale"],
                        )
                    # start ON the grid, so step 1 is not a special case
                    with torch.no_grad():
                        p.copy_(
                            quantize(
                                p,
                                spec=shadow_spec,
                                fixed_scale=st["fixed_scale"],
                            ).q
                        )
                idx += 1

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None

        frozen_num = frozen_den = 0
        grad_zero_num = grad_zero_den = 0.0
        sat_num = 0.0
        n_tensors = 0

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                g = p.grad

                if self.grad_spec.kind != "none":
                    gr = quantize(g, spec=self.grad_spec, generator=st["grad_gen"])
                    if not self.grad_quant_changed_ever:
                        self.grad_quant_changed_ever = bool((gr.q != g).any().item())
                    g = gr.q
                    grad_zero_num += gr.stats["zero_frac"]
                    grad_zero_den += 1.0

                if momentum:
                    buf = st.get("momentum_buffer")
                    if buf is None:
                        buf = st["momentum_buffer"] = torch.zeros_like(p)
                    buf.mul_(momentum).add_(g)
                    g = buf

                target = p - lr * g

                if self.shadow_spec.kind == "none":
                    new = target
                elif self.error_feedback:
                    new = st["ef"].apply(target)
                    sat_num += st["ef"].last_stats.get("sat_frac", 0.0)
                    n_tensors += 1
                else:
                    r = quantize(
                        target,
                        spec=self.shadow_spec,
                        generator=st["gen"],
                        fixed_scale=st["fixed_scale"],
                    )
                    new = r.q
                    sat_num += r.stats["sat_frac"]
                    n_tensors += 1

                if self.shadow_spec.kind != "none" and not self.shadow_changed_ever:
                    self.shadow_changed_ever = bool((new != target).any().item())
                frozen_num += int((new == p).sum().item())
                frozen_den += p.numel()
                p.copy_(new)

        self.last = {
            # fraction of coordinates the update failed to move: the stall canary
            "frozen_frac": frozen_num / frozen_den if frozen_den else 0.0,
            "grad_zero_frac": grad_zero_num / grad_zero_den if grad_zero_den else 0.0,
            "shadow_sat_frac": sat_num / n_tensors if n_tensors else 0.0,
        }
        return loss

    def assert_quantizers_active(self) -> None:
        """Raise if a configured optimizer-side quantizer never changed anything.

        The model-side analogue is quant.registry.assert_no_noops(). Without
        this, a shadow rung wired up wrongly would produce a clean FP32 loss
        curve labelled "8-bit latent weights" -- and R5 is the rung the whole
        project turns on. Call after a warm-up step.
        """
        bad = []
        if self.shadow_spec.kind != "none" and not self.shadow_changed_ever:
            bad.append(
                f"shadow weights ({self.shadow_spec.describe()}) never changed the "
                f"update; delta may be far below the update size, or the spec is inert"
            )
        if self.grad_spec.kind != "none" and not self.grad_quant_changed_ever:
            bad.append(
                f"gradient quantizer ({self.grad_spec.describe()}) never changed a gradient"
            )
        if bad:
            raise AssertionError("silent no-op in IntSGD: " + "; ".join(bad))

    def freeze_threshold(self, lr: float) -> float:
        """Per-coordinate freeze threshold on |lr * g_i|, namely D/2.

        Under round-to-nearest, w_i is on the grid, so
            Q(w_i - lr*g_i) = w_i  <=>  |lr*g_i| < D/2
        exactly. Coordinates above the threshold do not freeze; they limit-cycle.
        No free parameters -- see the module docstring.
        """
        if self.delta is None:
            return 0.0
        return self.delta / 2.0


def quantize_grad(g: torch.Tensor, spec: QuantSpec, generator=None) -> torch.Tensor:
    """Standalone gradient quantizer, exposed for tests and the audit."""
    return fake_quant(g, spec=spec, generator=generator)
