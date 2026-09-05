"""nanoGPT-style decoder with swappable QuantLinear.

Sized so the TRANSFORMER BODY dominates. At M2 the first MLP design put 86% of
its parameters outside the ladder and therefore measured nothing; the same trap
is far worse for a transformer. With a 50k GPT-2 vocab and d_model=192 the
embedding table alone would be 9.6M parameters against a 2.7M body, so every
rung would really be measuring a lookup table. A 2048-token BPE inverts that:
~86% of parameters sit inside the ladder.

Attention is written out explicitly rather than calling
F.scaled_dot_product_attention, because the two activation-activation matmuls
(QK^T and AV) are themselves rungs at R6. SDPA would hide them inside a fused
kernel where they cannot be quantized or even counted.

Known FP operations still present at R0, i.e. the M4 audit target list:
  - LayerNorm: mean, variance, rsqrt, and an elementwise affine multiply
  - softmax: exp and divide forward; two elementwise multiplies backward
  - attention scale 1/sqrt(head_dim)  (a multiply; pow2 head_dim makes it a shift)
  - GELU
  - residual adds (free only if scales are aligned)
  - the embedding gather (free) and the tied LM head matmul (not free)
  - cross-entropy: log-softmax and mean
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.quant_mlp import LadderConfig, QuantLinear

NONE_LADDER = LadderConfig()


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    ctx: int = 128
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    ladder: LadderConfig = field(default_factory=LadderConfig)
    quantize_embedding: bool = False   # BitNet-family papers keep these in FP
    tie_weights: bool = True
    act: str = "gelu"                  # "gelu" | "relu" (relu is R7, multiplier-free)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = QuantLinear(cfg.d_model, 3 * cfg.d_model, cfg.ladder,
                               name=f"blk{layer}.qkv", seed=seed)
        self.proj = QuantLinear(cfg.d_model, cfg.d_model, cfg.ladder,
                                name=f"blk{layer}.attn_proj", seed=seed)
        mask = torch.tril(torch.ones(cfg.ctx, cfg.ctx)).view(1, 1, cfg.ctx, cfg.ctx)
        self.register_buffer("mask", mask, persistent=False)
        self.scale = 1.0 / math.sqrt(cfg.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)

        # activation x activation matmul #1 -- an R6 target, both operands dynamic
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # activation x activation matmul #2 -- an R6 target
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int, seed: int) -> None:
        super().__init__()
        self.fc = QuantLinear(cfg.d_model, 4 * cfg.d_model, cfg.ladder,
                              name=f"blk{layer}.mlp_fc", seed=seed)
        self.proj = QuantLinear(4 * cfg.d_model, cfg.d_model, cfg.ladder,
                                name=f"blk{layer}.mlp_proj", seed=seed)
        # deliberately NOT gated: a gated MLP multiplies two activations, which
        # is a data-dependent multiply no scaling scheme can turn into a shift
        self.act = F.gelu if cfg.act == "gelu" else F.relu

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.act(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int, seed: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer, seed)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, layer, seed)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class QuantGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, i, seed) for i in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)
        # scaled init on residual projections (GPT-2 recipe)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, QuantLinear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_breakdown(self) -> dict:
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        head = 0 if self.cfg.tie_weights else self.head.weight.numel()
        body = total - emb - head
        return {
            "total": total,
            "embedding": emb,
            "head": head,
            "body": body,
            "body_frac": body / total,
        }
