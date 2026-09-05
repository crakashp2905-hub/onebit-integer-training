"""Error feedback (residual accumulation) for quantization.

The alternative to stochastic rounding, and probably the more important one for
latent weights.

The problem both solve: if a quantizer has grid step D and you repeatedly
quantize a value whose magnitude is below D/2, round-to-nearest returns zero
EVERY time and the information is annihilated. This is "swamping", and it is the
predicted failure mode of quantized shadow weights (rung R5) -- a stall, not a
divergence.

    stochastic rounding   unbiased per step, variance D^2 * f(1-f) per step.
                          Correct in expectation, but it turns the accumulator
                          into a random walk, so error grows like sqrt(T).

    error feedback        keep the residual and add it back next step:
                              v = x + e,  q = Q(v),  e <- v - q
                          Biased on any single step, but the ACCUMULATED bias is
                          zero by construction: sum_t q_t = sum_t x_t - e_T with
                          |e_T| <= D/2 forever. Error is bounded, not growing.

Error feedback is what made 1-bit SGD work (Seide et al. 2014) and has
convergence theory behind it (Karimireddy et al. 2019), both in the context of
compressing GRADIENTS. Applying it to the latent weight itself is the natural
extension, and is the main thing we will test at R5.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import QuantSpec
from .quantizers import quantize


class ErrorFeedback:
    """Stateful quantizer that carries its rounding residual forward.

    One instance per tensor being quantized (the residual has the tensor shape).
    Not an nn.Module: the state lives here so it is obvious what is stored and
    how much of it there is.
    """

    def __init__(
        self,
        spec: QuantSpec,
        *,
        generator: torch.Generator | None = None,
        fixed_scale: Tensor | None = None,
    ) -> None:
        self.spec = spec
        self.generator = generator
        self.fixed_scale = fixed_scale
        self.residual: Tensor | None = None
        self.last_stats: dict[str, float] = {}
        self.calls = 0

    def apply(self, x: Tensor) -> Tensor:
        """Quantize x + residual, store the new residual, return the quantized value."""
        if self.spec.kind == "none":
            return x
        if self.residual is None:
            self.residual = torch.zeros_like(x)
        elif self.residual.shape != x.shape:
            raise ValueError(
                "ErrorFeedback residual shape "
                + str(tuple(self.residual.shape))
                + " does not match input "
                + str(tuple(x.shape))
            )
        v = x + self.residual
        res = quantize(
            v, spec=self.spec, generator=self.generator, fixed_scale=self.fixed_scale
        )
        self.residual = v - res.q
        self.calls += 1
        self.last_stats = dict(res.stats)
        self.last_stats["residual_absmax"] = self.residual.abs().max().item()
        nz = x != 0
        n = int(nz.sum().item())
        self.last_stats["annihilated_frac"] = (
            float(((res.q == 0) & nz).sum().item()) / n if n else 0.0
        )
        return res.q

    def reset(self) -> None:
        self.residual = None
        self.last_stats = {}
        self.calls = 0
