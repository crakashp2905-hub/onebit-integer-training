"""Quantize / dequantize / fake-quantize.

Three separate functions on purpose:

    quantize(x)              -> integer CODES + scale   (what Track B consumes)
    dequantize(codes, scale) -> float values on the grid
    fake_quant(x)            -> float values on the grid (what the model consumes)

Track A never executes integer arithmetic, so it is easy to write a "quantizer"
that silently isn't one. Keeping the integer codes as a real, inspectable output
-- and asserting dequantize(quantize(x)) == fake_quant(x) bit-exactly -- is what
keeps the simulation honest about the integer pipeline it stands in for.

Known ways Track A differs from a real integer kernel (record these; do not
forget them when interpreting results):
  - accumulation happens in FP32 here, whereas INT8xINT8->INT32 accumulation is
    EXACT until overflow. Track A is therefore PESSIMISTIC about accumulation.
  - the scale arithmetic itself (x/scale, codes*scale) runs at full FP32
    precision here. A real kernel must do this in fixed point. Track A is
    OPTIMISTIC about requantization, which is exactly why scale_mode is an
    explicit, measurable axis.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch import Tensor

from .config import QuantSpec
from .round import apply_rounding
from .scale import compute_scale, group_view


class QuantResult(NamedTuple):
    q: Tensor          # dequantized values, original shape -- what the model uses
    codes: Tensor      # integer codes, original shape, narrowest int dtype
    scale: Tensor      # positive, broadcasts against the 2-D group view
    spec: QuantSpec    # recorded verbatim so results rows are self-describing
    stats: dict[str, Any]


def codes_dtype(spec: QuantSpec) -> torch.dtype:
    """Narrowest integer dtype that can hold this spec's codes."""
    qmax = spec.qmax
    if qmax <= 127:
        return torch.int8
    if qmax <= 32767:
        return torch.int16
    return torch.int32


def quantize(
    x: Tensor,
    *,
    spec: QuantSpec,
    generator: torch.Generator | None = None,
    fixed_scale: Tensor | None = None,
) -> QuantResult:
    """Quantize x under spec.

    fixed_scale overrides calibration. This matters for shadow weights (R5),
    where the latent grid must be FIXED: a dynamically calibrated scale would
    silently rescale itself to whatever the weights currently are, which hides
    the very swamping effect we are trying to measure.
    """
    if spec.kind == "none":
        return QuantResult(
            q=x,
            codes=x,
            scale=torch.ones((), dtype=x.dtype, device=x.device),
            spec=spec,
            stats={"passthrough": True, "zero_frac": 0.0, "sat_frac": 0.0, "n_groups": 1},
        )

    if not x.is_floating_point():
        raise TypeError(f"quantize expects a floating tensor, got {x.dtype}")

    qmax = spec.qmax

    if fixed_scale is None:
        view, scale, _ = compute_scale(x, spec)
    else:
        view, _ = group_view(x, spec)
        scale = fixed_scale.to(dtype=view.dtype, device=view.device)
        if scale.numel() != 1:
            expected = list(view.shape)
            expected[1 if spec.granularity != "col" else 0] = 1
            if list(scale.shape) != expected:
                raise ValueError(
                    f"fixed_scale shape {tuple(scale.shape)} does not match the group "
                    f"view for spec {spec.describe()} (expected {tuple(expected)} or scalar)"
                )
        scale = scale.clamp_min(torch.finfo(view.dtype).tiny)

    pre = view / scale
    rounded = apply_rounding(pre, spec.rounding, generator)
    saturated = rounded.abs() > qmax          # the clamp below actually changed something
    codes_f = rounded.clamp_(-qmax, qmax)

    q = (codes_f * scale).reshape(x.shape)
    codes = codes_f.to(codes_dtype(spec)).reshape(x.shape)

    stats = {
        "passthrough": False,
        "zero_frac": (codes_f == 0).to(torch.float32).mean().item(),
        "sat_frac": saturated.to(torch.float32).mean().item(),
        "scale_min": scale.min().item(),
        "scale_max": scale.max().item(),
        "n_groups": scale.numel(),
    }
    return QuantResult(q=q, codes=codes, scale=scale, spec=spec, stats=stats)


def dequantize(codes: Tensor, scale: Tensor, spec: QuantSpec) -> Tensor:
    """codes * scale, with the grouping implied by spec.

    Takes spec (rather than only codes and scale) because the scale is stored in
    group-view shape, not broadcast to every element. That is a deliberate memory
    choice: a per-element scale for a block-quantized 25M-parameter tensor would
    be another 100 MB of float32 for no information gain.
    """
    if spec.kind == "none":
        return codes
    view, _ = group_view(codes.to(scale.dtype), spec)
    return (view * scale).reshape(codes.shape)


def fake_quant(
    x: Tensor,
    *,
    spec: QuantSpec,
    generator: torch.Generator | None = None,
    fixed_scale: Tensor | None = None,
) -> Tensor:
    """Values snapped to the low-precision grid, still in floating point."""
    return quantize(x, spec=spec, generator=generator, fixed_scale=fixed_scale).q


# ---------------------------------------------------------------- named recipes

def quantize_ternary(
    w: Tensor,
    *,
    spec: QuantSpec | None = None,
    generator: torch.Generator | None = None,
) -> QuantResult:
    """BitNet b1.58 weight quantization.

        beta  = mean(|W|)                       (spec.calib="absmean")
        codes = clamp(round(W / beta), -1, +1)
        W_q   = codes * beta

    Defaults to the published recipe: per-tensor, absmean, round-to-nearest.
    Every part of it is overridable through spec so the ladder can vary one
    thing at a time.
    """
    if spec is None:
        spec = QuantSpec(kind="ternary", granularity="tensor", calib="absmean")
    if spec.kind != "ternary":
        raise ValueError(f"quantize_ternary needs kind='ternary', got {spec.kind!r}")
    return quantize(w, spec=spec, generator=generator)


def quantize_int(
    x: Tensor,
    *,
    spec: QuantSpec,
    generator: torch.Generator | None = None,
    fixed_scale: Tensor | None = None,
) -> QuantResult:
    """Symmetric int-N quantization.

        scale = calib(|x|) / (2^(bits-1) - 1)
        codes = clamp(round(x / scale), -qmax, +qmax)

    The BitNet activation recipe is bits=8, granularity="row" (per token),
    calib="absmax".
    """
    if spec.kind != "int":
        raise ValueError(f"quantize_int needs kind='int', got {spec.kind!r}")
    return quantize(x, spec=spec, generator=generator, fixed_scale=fixed_scale)
