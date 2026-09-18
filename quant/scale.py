"""Scale computation and scale-representation constraints.

Two separate things live here and they must not be confused:

  1. CALIBRATION -- how large is this group of numbers? (absmax / absmean /
     percentile). This determines the grid step.

  2. REPRESENTATION -- what is the scale allowed to BE? A free float scale means
     the requantization at the end of every matmul is a floating-point multiply.
     Constraining the scale to a power of two turns that multiply into a shift,
     which is what "multiplier-free" actually requires. This axis is independent
     of bit-width and is expected to be a binding constraint in its own right.

All grouping is expressed through a 2-D "group view": every quantizer reshapes
its input to 2-D, reduces along one axis with keepdim, and reshapes back. This
keeps scales small (never materialized per-element) and keeps the code boring.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import QuantSpec

_SQRT_HALF = math.sqrt(0.5)  # geometric midpoint of [2^(e-1), 2^e] in mantissa terms


def group_view(x: Tensor, spec: QuantSpec) -> tuple[Tensor, int]:
    """Reshape x to 2-D so that one reduction axis defines the groups.

    Returns (view, reduce_dim). The scale computed as
    ``view.abs().amax(reduce_dim, keepdim=True)`` broadcasts against ``view``.

    granularity:
      "tensor"  one scale for everything
      "row"     one scale per row of the LAST dim  (weights: per output channel;
                activations [B,T,D]: per token -- this is the BitNet convention)
      "col"     one scale per column, 2-D tensors only (per input channel)
      "block"   one scale per contiguous run of block_size elements in the
                FLATTENED tensor (the MX / bitsandbytes convention)
    """
    g = spec.granularity
    if g == "tensor":
        return x.reshape(1, -1), 1
    if g == "row":
        if x.dim() < 1 or x.shape[-1] == 0:
            raise ValueError(f"granularity='row' needs a non-empty last dim; got {tuple(x.shape)}")
        return x.reshape(-1, x.shape[-1]), 1
    if g == "col":
        if x.dim() != 2:
            raise ValueError(f"granularity='col' is 2-D only; got shape {tuple(x.shape)}")
        return x, 0
    if g == "block":
        if x.numel() % spec.block_size != 0:
            raise ValueError(
                f"block_size={spec.block_size} does not divide numel={x.numel()}"
            )
        return x.reshape(-1, spec.block_size), 1
    raise ValueError(f"unknown granularity {g!r}")


def calibrate(view: Tensor, reduce_dim: int, spec: QuantSpec) -> Tensor:
    """Per-group magnitude statistic, shape broadcastable against ``view``."""
    a = view.abs()
    if spec.calib == "absmax":
        return a.amax(dim=reduce_dim, keepdim=True)
    if spec.calib == "absmean":
        return a.mean(dim=reduce_dim, keepdim=True)
    if spec.calib == "percentile":
        # torch.quantile caps the reduced dimension at ~16M elements.
        q = spec.percentile / 100.0
        return torch.quantile(a.to(torch.float32), q, dim=reduce_dim, keepdim=True).to(a.dtype)
    raise ValueError(f"unknown calib {spec.calib!r}")


def to_pow2(s: Tensor, mode: str = "ceil") -> Tensor:
    """Round a positive scale to a power of two, EXACTLY.

    Uses frexp rather than log2/exp2 so the result is bit-exact and idempotent:
    a scale that is already 2^k is returned unchanged.

    mode="ceil"   smallest 2^k >= s. Guarantees nothing saturates, at a cost of
                  up to one bit of resolution. Safe default for gradients.
    mode="round"  nearest 2^k in the GEOMETRIC sense (midpoint at sqrt(2)).
                  Better average resolution, but can saturate.
    """
    m, e = torch.frexp(s)  # s = m * 2^e, m in [0.5, 1) for s > 0
    if mode == "ceil":
        exp = torch.where(m > 0.5, e, e - 1)
    elif mode == "round":
        exp = torch.where(m > _SQRT_HALF, e, e - 1)
    else:
        raise ValueError(f"unknown pow2 mode {mode!r}")
    out = torch.ldexp(torch.ones_like(s), exp)
    return torch.where(s > 0, out, s)  # leave zeros/denormal-floor alone


def to_dyadic(s: Tensor, mantissa_bits: int) -> Tensor:
    """Constrain a positive scale to M * 2^k with M having mantissa_bits bits.

    This is NOT multiplier-free -- it needs a small integer multiply plus a
    shift -- but it is what integer-only inference kernels actually do
    (dyadic requantization). The point of exposing mantissa_bits is to measure
    how many bits of scale precision are needed, i.e. the price of going all
    the way to pure shifts (mantissa_bits=0).
    """
    if mantissa_bits < 0:
        raise ValueError("mantissa_bits must be >= 0")
    m, e = torch.frexp(s)          # s = m * 2^e,  m in [0.5, 1)
    mm = m * 2.0                   # mm in [1, 2)
    ee = e - 1                     # s = mm * 2^ee
    step = float(2**mantissa_bits)
    mq = torch.round(mm * step) / step
    out = torch.ldexp(mq, ee)
    return torch.where(s > 0, out, s)


def apply_scale_mode(s: Tensor, spec: QuantSpec) -> Tensor:
    if spec.scale_mode == "float":
        return s
    if spec.scale_mode == "pow2":
        return to_pow2(s, spec.pow2_mode)
    if spec.scale_mode == "dyadic":
        return to_dyadic(s, spec.mantissa_bits)
    raise ValueError(f"unknown scale_mode {spec.scale_mode!r}")


def compute_scale(x: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor, int]:
    """Full scale pipeline.

    Returns (view, scale, reduce_dim) where ``scale`` broadcasts against ``view``.

    The scale is floored at finfo.tiny so an all-zero group cannot produce a
    division by zero. A group whose magnitude is small enough to hit that floor
    quantizes to all zeros -- a real underflow, reported by the zero_frac
    statistic rather than hidden.
    """
    view, reduce_dim = group_view(x, spec)
    amax = calibrate(view, reduce_dim, spec)
    scale = amax / spec.qmax * spec.scale_mult
    scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
    scale = apply_scale_mode(scale, spec)
    return view, scale, reduce_dim
