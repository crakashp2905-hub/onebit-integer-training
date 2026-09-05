"""Rounding modes.

Stochastic rounding is the load-bearing primitive of this project, so its
contract is stated precisely:

    SR(x) = floor(x + u),  u ~ Uniform[0, 1)

Write x = n + f with n = floor(x) and f = frac(x) in [0, 1). Then
SR(x) = n with probability 1-f and n+1 with probability f, so

    E[SR(x)] = n(1-f) + (n+1)f = n + f = x          exactly unbiased
    Var[SR(x)] = f(1-f)                             <= 1/4

Two consequences used by the tests:
  - if x is an exact integer (f=0) then SR(x) = x with probability 1
  - SR(x) is always in {floor(x), ceil(x)}

The unbiasedness is what lets many small updates accumulate through a coarse
grid instead of being annihilated by round-to-nearest. The variance is what
turns a latent weight into a random walk. Both matter; see error_feedback.py
for the lower-variance alternative.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import Rounding

# dtypes we can safely draw uniforms in directly
_WIDE = (torch.float32, torch.float64)


def stochastic_round(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Unbiased stochastic rounding to integers. See module docstring.

    The generator is explicit and never the global RNG stream: stochastic
    rounding consumes a data-dependent number of random values, so drawing from
    the global stream would make batch order (and therefore whole training runs)
    irreproducible.

    For float16/bfloat16 inputs the uniform is drawn in float32 and cast, which
    makes u slightly discrete; unbiasedness is then approximate, not exact.
    """
    if generator is not None and generator.device.type != x.device.type:
        raise ValueError(
            f"generator is on {generator.device} but tensor is on {x.device}; "
            "pass a generator created on the tensor's device"
        )
    if x.dtype in _WIDE:
        u = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    else:
        u = torch.rand(
            x.shape, generator=generator, device=x.device, dtype=torch.float32
        ).to(x.dtype)
    return torch.floor(x + u)


def apply_rounding(
    x: Tensor, mode: Rounding, generator: torch.Generator | None = None
) -> Tensor:
    """Round to integers under the named mode.

    "nearest" is torch.round, i.e. round-half-to-EVEN, not half-away-from-zero.
    "floor" is a deliberately biased control: it exists so ablations can show
    that the bias, not the noise, is what breaks training.
    """
    if mode == "nearest":
        return torch.round(x)
    if mode == "stochastic":
        return stochastic_round(x, generator)
    if mode == "floor":
        return torch.floor(x)
    raise ValueError(f"unknown rounding mode {mode!r}")


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a dedicated generator. Sites should derive seeds from
    (global_seed, site_id, step) so runs stay reproducible."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
