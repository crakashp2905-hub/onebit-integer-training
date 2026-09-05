"""Straight-through estimators.

Why an STE is needed: quantization is piecewise constant, so its true derivative
is 0 almost everywhere and undefined at the steps. Backpropagating that gives
exactly zero gradient and no learning. The STE substitutes the derivative of a
DIFFERENT function (usually the identity) for the backward pass.

What the STE hides -- worth being precise about, because it is the assumption
this whole project leans on:

  1. The forward and backward passes correspond to different functions, so the
     "gradient" produced is not the gradient of anything. There is no descent
     guarantee and no objective actually being minimized.
  2. The mismatch is largest exactly where quantization error is largest, i.e.
     mid-bin, so the error is systematic rather than noise.
  3. With the plain identity STE, latent weights can drift far outside the
     quantizer range, where the forward pass is completely insensitive to them
     but the backward pass still moves them. The "clipped" variant zeroes the
     gradient outside +/- ste_clip to stop that drift.

ste="none" (true zero gradient) exists as a control: it lets the ladder measure
what the STE actually buys, rather than assuming it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import registry
from .config import QuantSpec
from .quantizers import quantize


class _STE(torch.autograd.Function):
    """forward: emit the pre-computed quantized tensor. backward: route to x."""

    @staticmethod
    def forward(ctx, x: Tensor, q: Tensor, mode: str, clip: float):  # type: ignore[override]
        ctx.mode = mode
        ctx.clip = clip
        if mode == "clipped":
            ctx.save_for_backward(x)
        return q

    @staticmethod
    def backward(ctx, g: Tensor):  # type: ignore[override]
        if ctx.mode == "identity":
            return g, None, None, None
        if ctx.mode == "clipped":
            (x,) = ctx.saved_tensors
            return g * (x.abs() <= ctx.clip).to(g.dtype), None, None, None
        if ctx.mode == "none":
            # the TRUE derivative of a step function
            return torch.zeros_like(g), None, None, None
        raise ValueError("unknown ste mode: " + str(ctx.mode))


def fake_quant_ste(
    x: Tensor,
    *,
    spec: QuantSpec,
    generator: torch.Generator | None = None,
    fixed_scale: Tensor | None = None,
    site: str | None = None,
) -> Tensor:
    """Quantize in the forward pass, pass gradient through in the backward pass.

    The quantization itself runs under no_grad so it contributes no autograd
    graph: the ONLY gradient path is the STE. Without this, the division and
    multiplication by the scale would silently create a second, non-STE path.

    Passing ``site`` records the call in the registry, which is what powers the
    runtime no-op assertion and the float-op audit.
    """
    if spec.kind == "none":
        return x
    with torch.no_grad():
        res = quantize(x, spec=spec, generator=generator, fixed_scale=fixed_scale)
    if site is not None:
        registry.record(site, x, res)
    return _STE.apply(x, res.q, spec.ste, spec.ste_clip)


class _QuantBackward(torch.autograd.Function):
    """Identity forward, quantize the gradient on the way back."""

    @staticmethod
    def forward(ctx, x: Tensor, spec: QuantSpec, generator, site):  # type: ignore[override]
        ctx.spec = spec
        ctx.generator = generator
        ctx.site = site
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: Tensor):  # type: ignore[override]
        if ctx.spec.kind == "none":
            return g, None, None, None
        with torch.no_grad():
            res = quantize(g, spec=ctx.spec, generator=ctx.generator)
        if ctx.site is not None:
            registry.record(ctx.site, g, res)
        return res.q, None, None, None


def quantize_backward(
    x: Tensor,
    *,
    spec: QuantSpec,
    generator: torch.Generator | None = None,
    site: str | None = None,
) -> Tensor:
    """Quantize the gradient flowing back through this point (rung R3).

    Placed on a layer's OUTPUT, this quantizes dy -- the operand shared by both
    backward matmuls:

        input-grad   dx = dy @ W_q       (W_q already quantized by R1)
        weight-grad  dW = dy^T @ x_q     (x_q already quantized by R2)

    so one call covers the dynamic operand of both. The weight gradient that
    comes OUT of the second matmul is a separate knob, applied in the optimizer
    via IntSGD(grad_spec=...), because wgrad has the longest accumulation chain
    and is expected to need more precision than dgrad.

    Gradients are always calibrated dynamically: their magnitude moves by orders
    of magnitude over training, and a fixed scale would silently underflow. This
    is the integer analogue of loss scaling.
    """
    if spec.kind == "none":
        return x
    return _QuantBackward.apply(x, spec, generator, site)
