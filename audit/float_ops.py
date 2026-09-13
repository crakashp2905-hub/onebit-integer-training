"""Count floating-point MULTIPLIES in a training step, by source.

This is the metric the project is named after and the one thing that was never
built. Loss numbers say what quantization COSTS; they say nothing about what it
BOUGHT. "Multiplier-free" is a claim about operations, and until you count the
operations you cannot check it.

What counts as a multiply here:

    matmul       mm / bmm / addmm / matmul  ->  M*N*K multiplies
    elementwise  mul / div                  ->  numel multiplies
    transcendental exp / log / rsqrt / sqrt / erf / tanh / pow
                 counted SEPARATELY -- these are not multiplies, they are worse,
                 and lumping them in would flatter the result

Adds are free (integer add/accumulate is the target), so they are counted only
for context.

IMPORTANT about Track A. This is simulated quantization: an INT8 matmul still
executes as an FP32 matmul here, so a raw op count would show no improvement at
all. What the counter reports instead is the count each op WOULD contribute in a
real integer kernel, decided by whether its operands were quantized:

    a matmul whose operands both came from a quantizer -> 0 FP multiplies
      (it becomes integer multiply-accumulate, or shift-add for ternary)
    its requantization scale                           -> 1 FP multiply per
      output element, UNLESS the scale is a power of two, in which case 0

That mapping is the whole point: it is what turns "we constrained the scales to
powers of two" into a number.

Attribution is by module, via forward hooks that push a label stack, so the
table says WHERE the remaining multiplies are rather than just how many.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

MATMUL_OPS = {
    aten.mm.default, aten.bmm.default, aten.addmm.default,
    aten.matmul.default, aten.linear.default,
}
MUL_OPS = {
    aten.mul.Tensor, aten.mul.Scalar, aten.div.Tensor, aten.div.Scalar,
    aten.mul_.Tensor, aten.div_.Tensor,
}
TRANSCENDENTAL_OPS = {
    aten.exp.default, aten.log.default, aten.rsqrt.default, aten.sqrt.default,
    aten.erf.default, aten.tanh.default, aten.pow.Tensor_Scalar,
    aten._softmax.default, aten._softmax_backward_data.default,
    aten._log_softmax.default, aten._log_softmax_backward_data.default,
}
ADD_OPS = {
    aten.add.Tensor, aten.add_.Tensor, aten.sub.Tensor, aten.sum.default,
    aten.sum.dim_IntList,
}


def _numel(x) -> int:
    return x.numel() if isinstance(x, torch.Tensor) else 0


def _matmul_cost(func, args) -> int:
    """M*N*K multiply-accumulates for a matmul, from the operand shapes."""
    ts = [a for a in args if isinstance(a, torch.Tensor)]
    if func is aten.addmm.default and len(ts) >= 3:
        a, b = ts[1], ts[2]
    elif len(ts) >= 2:
        a, b = ts[0], ts[1]
    else:
        return 0
    if a.dim() < 2 or b.dim() < 2:
        return 0
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    batch = 1
    for d in a.shape[:-2]:
        batch *= d
    return batch * m * n * k


@dataclass
class Bucket:
    fp_mul: int = 0          # multiplies that survive in a real integer kernel
    raw_mul: int = 0         # multiplies as literally executed in Track A
    transcendental: int = 0
    add: int = 0
    calls: collections.Counter = field(default_factory=collections.Counter)


class FloatOpCounter(TorchDispatchMode):
    """Count ops inside a `with` block, attributed to the current label.

    `quantized_matmul_depth` tracks whether we are inside a QuantLinear whose
    operands were quantized; matmuls there are scored as integer.
    """

    def __init__(self) -> None:
        super().__init__()
        # (label, operands_are_quantized, scales_are_pow2). A STACK, not two
        # scalars: attention contains QuantLinear children, so a child's pop
        # would otherwise clear the parent's flag and the QK^T / AV matmuls --
        # which run AFTER self.qkv returns -- would always score as full FP.
        # That silently zeroes the entire R6 benefit.
        self.stack: list[tuple[str, bool, bool]] = []
        self.buckets: dict[str, Bucket] = collections.defaultdict(Bucket)
        self._muted = 0

    # -- label stack ------------------------------------------------------
    def push(self, label: str, quant: bool = False, pow2: bool = False) -> None:
        self.stack.append((label, quant, pow2))

    def pop(self) -> None:
        if self.stack:
            self.stack.pop()

    @property
    def label(self) -> str:
        return self.stack[-1][0] if self.stack else "unattributed"

    @property
    def quant_matmul(self) -> bool:
        return self.stack[-1][1] if self.stack else False

    @property
    def pow2_scales(self) -> bool:
        return self.stack[-1][2] if self.stack else False

    def mute(self) -> "._Mute":
        return _Mute(self)

    # -- dispatch ---------------------------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self._muted:
            return out
        b = self.buckets[self.label]
        b.calls[str(func)] += 1

        if func in MATMUL_OPS:
            cost = _matmul_cost(func, args)
            b.raw_mul += cost
            if self.quant_matmul:
                # integer multiply-accumulate (or shift-add for ternary):
                # no FP multiply in the inner loop. The requantization scale
                # costs one FP multiply per OUTPUT element unless it is pow2.
                if not self.pow2_scales:
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    b.fp_mul += _numel(o)
            else:
                b.fp_mul += cost
        elif func in MUL_OPS:
            o = out[0] if isinstance(out, (tuple, list)) else out
            n = _numel(o)
            b.raw_mul += n
            b.fp_mul += n
        elif func in TRANSCENDENTAL_OPS:
            o = out[0] if isinstance(out, (tuple, list)) else out
            b.transcendental += _numel(o)
        elif func in ADD_OPS:
            o = out[0] if isinstance(out, (tuple, list)) else out
            b.add += _numel(o)
        return out

    # -- reporting --------------------------------------------------------
    def totals(self) -> dict[str, int]:
        return {
            "fp_mul": sum(b.fp_mul for b in self.buckets.values()),
            "raw_mul": sum(b.raw_mul for b in self.buckets.values()),
            "transcendental": sum(b.transcendental for b in self.buckets.values()),
            "add": sum(b.add for b in self.buckets.values()),
        }

    def by_source(self) -> list[tuple[str, Bucket]]:
        return sorted(self.buckets.items(), key=lambda kv: -kv[1].fp_mul)


class _Mute:
    """Suppress counting -- used for the quantizers' own scale arithmetic, which
    is Track A scaffolding and would not exist in a real integer kernel."""

    def __init__(self, counter: FloatOpCounter) -> None:
        self.c = counter

    def __enter__(self):
        self.c._muted += 1
        return self.c

    def __exit__(self, *exc):
        self.c._muted -= 1
        return False


def _matmul_is_integer(mod) -> tuple[bool, bool]:
    """Would this module's matmul carry FP multiplies in a real integer kernel?

    Returns (is_integer, scales_are_pow2).

    A TERNARY weight makes the matmul multiplier-free on its own, whatever the
    activation format: multiplying by {-1, 0, +1} is a select/negate/skip, not a
    multiply. That is the BitNet insight and it is why R1 alone already removes
    the bulk of the arithmetic.

    An INT weight needs an INT activation too, otherwise the product is still
    float x int and a real kernel must do an FP multiply.
    """
    attn = getattr(mod, "attn_spec", None)
    if attn is not None:
        # R6: QK^T and AV are activation x activation. There is no weight to
        # make ternary, so BOTH operands must be integer -- this is the one
        # place where the BitNet shortcut does not apply.
        if attn.kind != "int":
            return False, False
        return True, attn.scale_mode == "pow2" and getattr(mod, "attn_scale_pow2", False)

    w = getattr(mod, "w_spec", None)
    a = getattr(mod, "a_spec", None)
    if w is None:
        return False, False
    is_int = (w.kind == "ternary") or (w.kind == "int" and a is not None
                                       and a.kind == "int")
    if not is_int:
        return False, False
    specs = [x for x in (w, a) if x is not None and x.kind != "none"]
    pow2 = bool(specs) and all(x.scale_mode == "pow2" for x in specs)
    return True, pow2


def _backward_matmul_is_integer(mod) -> tuple[bool, bool]:
    """Same question for the TWO backward matmuls.

        dgrad = dy @ W    -- a ternary W makes this multiplier-free whatever dy
                             is; an int W additionally needs dy quantized
        wgrad = dy^T @ x  -- needs BOTH the saved activation and dy quantized

    Scored conservatively: the flag is per-label, not per-matmul, so it is set
    only when BOTH backward matmuls qualify. That can understate a config where
    dgrad is integer but wgrad is not (ternary weights with FP gradients, i.e.
    R1 and R2), never overstate one.

    Getting this wrong in the other direction was the first version's bug: the
    flag was set in the forward hook only, so every backward matmul scored as
    full floating point and the audit reported 26% elimination for a config
    whose backward is largely integer.
    """
    # attention backward (dQ, dK, dV) is NOT quantized by R6 as implemented,
    # so it is scored as full floating point. Understates R6, never overstates.
    if getattr(mod, "attn_spec", None) is not None:
        return False, False
    w = getattr(mod, "w_spec", None)
    a = getattr(mod, "a_spec", None)
    g = getattr(mod, "g_spec", None)
    if w is None:
        return False, False
    gq = g is not None and g.kind == "int"
    aq = a is not None and a.kind == "int"
    dgrad_int = (w.kind == "ternary" and gq) or (w.kind == "int" and gq)
    wgrad_int = aq and gq
    if not (dgrad_int and wgrad_int):
        return False, False
    specs = [x for x in (w, a, g) if x is not None and x.kind != "none"]
    return True, bool(specs) and all(x.scale_mode == "pow2" for x in specs)


#: modules that do their own arithmetic and must be labelled even though they
#: have children -- attention owns the two activation-activation matmuls (QK^T
#: and AV), which are R6 targets and would otherwise vanish into the parent.
_ALWAYS_LABEL = {"QuantLinear", "CausalSelfAttention"}


def attach_labels(model: torch.nn.Module, counter: FloatOpCounter) -> list:
    """Push/pop a module label around BOTH the forward and the backward pass.

    Forward hooks alone leave the backward unattributed, which for a training
    step is most of the work: measured 3.58e9 of 5.14e9 multiplies (70%) landing
    in "unattributed" before backward hooks were added. Since the whole point is
    to say WHERE the multiplies are, that is not good enough.

    Backward labels are suffixed so forward and backward costs stay separable --
    the three matmuls per linear (forward, dgrad, wgrad) are different rungs and
    were expected to need different precision.
    """
    handles = []
    for name, mod in model.named_modules():
        if not name:
            continue
        kind = type(mod).__name__
        if list(mod.children()) and kind not in _ALWAYS_LABEL:
            continue
        label = f"{name} [{kind}]"

        def _pre(m, i, lbl=label, c=counter):
            c.push(lbl, *_matmul_is_integer(m))

        def _post(m, i, o, c=counter):
            c.pop()

        handles.append(mod.register_forward_pre_hook(_pre))
        handles.append(mod.register_forward_hook(_post))
        def _bpre(m, go, lbl=label + " bwd", c=counter):
            c.push(lbl, *_backward_matmul_is_integer(m))

        def _bpost(m, gi, go, c=counter):
            c.pop()

        handles.append(mod.register_full_backward_pre_hook(_bpre))
        handles.append(mod.register_full_backward_hook(_bpost))
    return handles
