"""Quantization specifications.

A QuantSpec fully describes one quantization site. It is frozen and hashable so
it can be recorded verbatim in every result row (methodology requirement: every
config fully recorded with each result).

Vocabulary used throughout this package:

  grid step (delta)  the spacing between representable values, = scale
  scale              positive float multiplying integer codes to get values
  codes              the integers actually stored (int8/int16/int32)
  group              the set of elements sharing one scale (granularity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["none", "ternary", "int"]
Granularity = Literal["tensor", "row", "col", "block"]
ScaleMode = Literal["float", "pow2", "dyadic"]
Rounding = Literal["nearest", "stochastic", "floor"]
Calib = Literal["absmax", "absmean", "percentile"]
STEMode = Literal["identity", "clipped", "none"]


@dataclass(frozen=True)
class QuantSpec:
    """One quantization site's numerical configuration.

    kind
        "none"     pass-through (used as an ablation control)
        "ternary"  BitNet b1.58: codes in {-1,0,+1}, scale = mean|x| by default
        "int"      symmetric int-N: codes in [-(2^(b-1)-1), +(2^(b-1)-1)]

    granularity / block_size
        Which elements share a scale. "block" splits the FLATTENED tensor into
        contiguous runs of block_size (the MX / bitsandbytes convention);
        block_size must divide numel. "col" is 2-D only.

    scale_mode
        "float"   unconstrained scale -- needs an FP multiply at requantization
        "pow2"    scale = 2^k -- requantization is a SHIFT (multiplier-free)
        "dyadic"  scale = M * 2^k with M having mantissa_bits bits -- needs a
                  small INTEGER multiply. mantissa_bits=0 rounds the mantissa
                  to 1 bit; note this uses the ARITHMETIC midpoint (1.5) while
                  pow2_mode="round" uses the GEOMETRIC midpoint (sqrt(2)), so
                  the two are close but not identical.

        This axis is INDEPENDENT of bits and is the one that makes training
        "multiplier-free". Measuring its cost separately from bit-width is a
        primary goal of the project.

    pow2_mode
        "ceil"    scale >= calibrated scale, so nothing saturates; costs up to
                  1 bit of resolution. Safe default for gradients.
        "round"   nearest power of two; better average resolution, saturates.

    rounding
        "nearest"     round-half-to-even (torch.round)
        "stochastic"  floor(x + u), u ~ U[0,1). Unbiased.
        "floor"       toward -inf (diagnostic / deliberately biased control)

    ste
        "identity"  backward passes gradient through unchanged
        "clipped"   backward zeroes gradient where |x| > ste_clip
        "none"      backward returns zero (the TRUE derivative of a step
                    function; included so the ladder can measure what STE buys)
    """

    kind: Kind = "int"
    bits: int = 8
    granularity: Granularity = "tensor"
    block_size: int = 0
    scale_mode: ScaleMode = "float"
    pow2_mode: Literal["ceil", "round"] = "ceil"
    mantissa_bits: int = 0
    rounding: Rounding = "nearest"
    calib: Calib = "absmax"
    percentile: float = 99.9
    ste: STEMode = "identity"
    ste_clip: float = 1.0
    # Multiplies the calibrated scale BEFORE scale_mode is applied. Exists for
    # one control: pow2_mode="ceil" inflates a ternary scale x1.3-1.95, which
    # also makes the ternary weights sparser. scale_mult=1.5 with a FLOAT scale
    # reproduces the sparsity change without the representation change.
    scale_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.kind == "int":
            if self.bits < 2:
                raise ValueError(
                    f"kind='int' needs bits>=2 (symmetric int-1 has qmax=0); got {self.bits}. "
                    "Use kind='ternary' for 1.58-bit."
                )
            if self.bits > 31:
                raise ValueError(f"bits>31 overflows int32 codes; got {self.bits}")
        if self.granularity == "block" and self.block_size <= 0:
            raise ValueError("granularity='block' requires block_size>0")
        if self.granularity != "block" and self.block_size:
            raise ValueError("block_size is only meaningful for granularity='block'")
        if self.scale_mode == "dyadic" and self.mantissa_bits < 0:
            raise ValueError("mantissa_bits must be >= 0")
        if self.scale_mult <= 0:
            raise ValueError("scale_mult must be > 0")
        if not (0.0 < self.percentile <= 100.0):
            raise ValueError("percentile must be in (0, 100]")

    @property
    def qmax(self) -> int:
        """Largest magnitude an integer code may take."""
        if self.kind == "ternary":
            return 1
        if self.kind == "int":
            return 2 ** (self.bits - 1) - 1
        raise ValueError(f"qmax undefined for kind={self.kind!r}")

    @property
    def n_levels(self) -> int:
        """Number of distinct representable values within one group."""
        return 2 * self.qmax + 1

    def describe(self) -> str:
        if self.kind == "none":
            return "none"
        base = "ternary" if self.kind == "ternary" else f"int{self.bits}"
        gran = self.granularity + (f"{self.block_size}" if self.granularity == "block" else "")
        smode = self.scale_mode
        if self.scale_mode == "pow2":
            smode += f"/{self.pow2_mode}"
        elif self.scale_mode == "dyadic":
            smode += f"/m{self.mantissa_bits}"
        mult = f"|x{self.scale_mult:g}" if self.scale_mult != 1.0 else ""
        return f"{base}|{gran}|{smode}{mult}|{self.rounding}|{self.calib}|ste={self.ste}"
