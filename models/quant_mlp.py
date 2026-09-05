"""A tiny MLP with every quantization site exposed as a config flag.

One QuantLinear covers three of the four numerical components:

    R1  weights      fake_quant_ste on the weight        (ternary, BitNet recipe)
    R2  activations  fake_quant_ste on the input         (INT8 per-token absmax)
    R3a dgrad        quantize_backward on the output     (quantizes dy)

The fourth, R5 (shadow weights) plus R3b (weight gradients), lives in the
optimizer, because that is where latent state and the accumulated weight
gradient actually are.

Every site registers itself, so `registry.assert_no_noops()` after a warm-up
step catches the failure mode where a rung is "enabled" but silently does
nothing -- the single most likely way this project produces a beautiful,
meaningless FP32 loss curve.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quant import QuantSpec, fake_quant_ste, make_generator, quantize_backward

NONE = QuantSpec(kind="none")


@dataclass
class LadderConfig:
    """Which rungs are active. Every field defaults to off, so a config is a
    literal description of the rungs under test."""

    name: str = "R0_fp32"
    w_spec: QuantSpec = NONE      # R1  weights
    a_spec: QuantSpec = NONE      # R2  activations
    g_spec: QuantSpec = NONE      # R3a dgrad (dy)
    wg_spec: QuantSpec = NONE     # R3b weight gradient (applied in the optimizer)
    shadow_bits: int | None = None      # R5  latent weight
    shadow_mode: str = "sr"             # rtn | sr | ef
    shadow_range: float = 4.0
    quantize_first_last: bool = False   # BitNet-family papers usually do NOT

    def describe(self) -> str:
        parts = []
        for tag, s in (("w", self.w_spec), ("a", self.a_spec),
                       ("dy", self.g_spec), ("wg", self.wg_spec)):
            if s.kind != "none":
                parts.append(f"{tag}={s.describe()}")
        if self.shadow_bits:
            parts.append(f"shadow={self.shadow_bits}b/{self.shadow_mode}")
        return self.name + (" [" + " ".join(parts) + "]" if parts else " [none]")


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: LadderConfig,
        *,
        name: str,
        seed: int = 0,
        quantize: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # a layer excluded from the ladder (first/last) runs in full precision
        self.w_spec = cfg.w_spec if quantize else NONE
        self.a_spec = cfg.a_spec if quantize else NONE
        self.g_spec = cfg.g_spec if quantize else NONE

        # zlib.crc32, NOT hash(): Python string hashing is randomized per
        # process unless PYTHONHASHSEED is set, which would silently break
        # cross-run reproducibility of every stochastic-rounding site.
        h = zlib.crc32(name.encode()) % 10_000
        self._base_seed = seed * 7919 + h
        # Generators are created LAZILY, per device. torch.Generator cannot be
        # moved across devices, so building them here would pin them to CPU;
        # calling model.to("cuda") would then move the parameters but not the
        # generators, and stochastic rounding would raise on the device check.
        # Not an nn.Module buffer: generators are RNG state, not model state.
        self._gens: dict[tuple[str, str, int], torch.Generator] = {}

    def _gen(self, kind: str, device: torch.device) -> torch.Generator:
        key = (kind, device.type, device.index if device.index is not None else -1)
        g = self._gens.get(key)
        if g is None:
            offset = {"w": 0, "a": 1, "g": 2}[kind]
            g = make_generator(self._base_seed + offset, device)
            self._gens[key] = g
        return g

    def forward(self, x: Tensor) -> Tensor:
        xq = fake_quant_ste(x, spec=self.a_spec, generator=self._gen("a", x.device),
                            site=f"{self.name}.act")
        wq = fake_quant_ste(self.weight, spec=self.w_spec,
                            generator=self._gen("w", self.weight.device),
                            site=f"{self.name}.w")
        y = F.linear(xq, wq, self.bias)
        return quantize_backward(y, spec=self.g_spec,
                                 generator=self._gen("g", y.device),
                                 site=f"{self.name}.dy")


class QuantMLP(nn.Module):
    """784 -> hidden -> ... -> 10, ReLU.

    ReLU deliberately, not a gated activation: a gated MLP multiplies two
    activations together, which is a data-dependent multiply that no scaling
    scheme can turn into a shift. Keeping the multiplier-free target honest
    starts at the architecture.
    """

    def __init__(
        self,
        cfg: LadderConfig,
        sizes: tuple[int, ...] = (784, 256, 128, 10),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        layers = []
        n = len(sizes) - 1
        for i in range(n):
            is_edge = (i == 0) or (i == n - 1)
            layers.append(
                QuantLinear(
                    sizes[i], sizes[i + 1], cfg,
                    name=f"fc{i}", seed=seed,
                    quantize=cfg.quantize_first_last or not is_edge,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
