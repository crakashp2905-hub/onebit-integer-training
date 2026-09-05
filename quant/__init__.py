"""Track A quantization primitives (simulated / fake quant)."""

from . import registry
from .config import QuantSpec
from .error_feedback import ErrorFeedback
from .quantizers import (
    QuantResult,
    codes_dtype,
    dequantize,
    fake_quant,
    quantize,
    quantize_int,
    quantize_ternary,
)
from .round import apply_rounding, make_generator, stochastic_round
from .scale import (
    apply_scale_mode,
    calibrate,
    compute_scale,
    group_view,
    to_dyadic,
    to_pow2,
)
from .ste import fake_quant_ste, quantize_backward

__all__ = [
    "QuantSpec",
    "QuantResult",
    "ErrorFeedback",
    "registry",
    "quantize",
    "dequantize",
    "fake_quant",
    "fake_quant_ste",
    "quantize_backward",
    "quantize_int",
    "quantize_ternary",
    "codes_dtype",
    "stochastic_round",
    "apply_rounding",
    "make_generator",
    "compute_scale",
    "calibrate",
    "group_view",
    "to_pow2",
    "to_dyadic",
    "apply_scale_mode",
]
