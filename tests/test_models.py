"""M2/M3 tests: the quantized MLP and the small decoder.

RECONSTRUCTED after the working tree was deleted.

Two tests here earned their place by catching real bugs, and both are kept
verbatim in spirit:

  - the causal-mask test. A mask that leaks future tokens makes every loss
    number in the project silently too good, and nothing else would notice.
  - the crc32-vs-hash subprocess test. Python string hashing is randomized per
    process; deriving generator seeds from it broke cross-run reproducibility.
    An earlier version of that test read subprocess stdout WITHOUT checking
    returncode, so when a rename broke the attribute it read, all three runs
    errored, stdout was empty for all three, and {""} has length 1 -- it passed
    while measuring nothing. It now asserts the subprocess actually succeeded.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

from data.mnist import frozen_batches
from models import (
    GPTConfig,
    LadderConfig,
    QuantGPT,
    QuantLinear,
    QuantMLP,
    count_params,
)
from quant import QuantSpec, registry

TERNARY = QuantSpec(kind="ternary", granularity="tensor", calib="absmean")
ACT8 = QuantSpec(kind="int", bits=8, granularity="row", calib="absmax")
DY8 = QuantSpec(kind="int", bits=8, granularity="row", rounding="stochastic")


@pytest.fixture(autouse=True)
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def x_batch(n=32, d=784, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def toks(b=4, t=16, v=128, seed=0):
    return torch.randint(0, v, (b, t), generator=torch.Generator().manual_seed(seed))


def small_cfg(ladder=None):
    return GPTConfig(vocab_size=128, ctx=16, n_layer=2, n_head=2, d_model=32,
                     ladder=ladder or LadderConfig())


# =========================================================== QuantLinear / MLP

def test_fp32_config_is_exactly_a_linear_layer():
    """The R0 arm must be a plain Linear, or every comparison is confounded."""
    layer = QuantLinear(16, 8, LadderConfig(), name="fc0")
    x = x_batch(4, 16)
    assert torch.equal(layer(x), F.linear(x, layer.weight, layer.bias))


def test_each_rung_changes_what_it_should():
    x = x_batch(8, 64)
    ternary = QuantLinear(64, 32, LadderConfig(w_spec=TERNARY), name="fc0")
    assert not torch.allclose(ternary(x), F.linear(x, ternary.weight, ternary.bias))
    act = QuantLinear(64, 32, LadderConfig(a_spec=ACT8), name="fc0")
    assert not torch.equal(act(x), F.linear(x, act.weight, act.bias))

    outs = {}
    for name, cfg in (("plain", LadderConfig()), ("dy8", LadderConfig(g_spec=DY8))):
        torch.manual_seed(0)
        layer = QuantLinear(64, 32, cfg, name="fc0")
        xin = x.clone().requires_grad_(True)
        layer(xin).pow(2).sum().backward()
        outs[name] = (xin.grad.clone(), layer.weight.grad.clone())
    assert not torch.equal(outs["plain"][0], outs["dy8"][0]), "dgrad unchanged"
    assert not torch.equal(outs["plain"][1], outs["dy8"][1]), "wgrad unchanged"


def test_gradients_reach_every_parameter_through_the_ste():
    model = QuantMLP(LadderConfig(w_spec=TERNARY, a_spec=ACT8, g_spec=DY8),
                     sizes=(64, 32, 32, 10))
    F.cross_entropy(model(x_batch(8, 64)), torch.zeros(8, dtype=torch.long)).backward()
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        if "weight" in n:
            assert p.grad.abs().sum() > 0, f"{n} gradient is all zero"


def test_excluding_edges_leaves_most_parameters_unquantized_in_a_shallow_net():
    """Documents the M2 design bug: with 3 layers the ladder touches only 14% of
    parameters, the full stack scored 0.9662 vs 0.9703 fp32 -- inside noise --
    and the experiment could not break anything, so it measured nothing."""
    cfg = LadderConfig(w_spec=TERNARY)
    shallow = QuantMLP(cfg, sizes=(784, 256, 128, 10))
    q = sum(l.weight.numel() for l in shallow.layers if l.w_spec.kind != "none")
    assert q / count_params(shallow) < 0.20
    deep = QuantMLP(cfg, sizes=(784, 256, 256, 256, 10))
    q2 = sum(l.weight.numel() for l in deep.layers if l.w_spec.kind != "none")
    assert q2 / count_params(deep) > 0.35


def test_registry_records_sites_and_the_guard_fires_when_one_is_inert():
    model = QuantMLP(LadderConfig(w_spec=TERNARY, a_spec=ACT8, g_spec=DY8),
                     sizes=(64, 32, 32, 10))
    model(x_batch(8, 64)).sum().backward()
    names = {r["site"] for r in registry.dump_stats()}
    assert any(n.endswith(".w") for n in names)
    assert any(n.endswith(".act") for n in names)
    assert any(n.endswith(".dy") for n in names)
    registry.assert_no_noops()
    for st in registry.sites().values():
        st.changed_ever = False
    with pytest.raises(AssertionError, match="silent no-op"):
        registry.assert_no_noops()


# ============================================================== determinism

def test_same_seed_gives_identical_output():
    cfg = LadderConfig(w_spec=TERNARY, a_spec=ACT8, g_spec=DY8)
    x = x_batch(8, 64)
    outs = []
    for _ in range(2):
        torch.manual_seed(3)
        outs.append(QuantMLP(cfg, sizes=(64, 32, 32, 10), seed=3)(x))
    assert torch.equal(outs[0], outs[1])


def test_site_generator_seeds_do_not_depend_on_python_hash_randomization():
    """Python string hashing is randomized per process. Measured across three
    processes before the fix: 8229 / 2383 / 8351. Every stochastic-rounding site
    would have been irreproducible across runs."""
    code = (
        "import sys, torch; sys.path.insert(0, r'%s');"
        "from models import QuantLinear, LadderConfig;"
        "l = QuantLinear(8, 4, LadderConfig(), name='fc1', seed=0);"
        "print(l._gen('w', torch.device('cpu')).initial_seed())" % str(ROOT)
    )
    seeds = set()
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        # MUST check returncode -- see the module docstring for why.
        assert r.returncode == 0, f"subprocess failed: {r.stderr[-400:]}"
        assert r.stdout.strip(), "subprocess produced no output"
        seeds.add(r.stdout.strip())
    assert len(seeds) == 1, f"generator seed varies across processes: {seeds}"


def test_stochastic_grad_quantization_does_not_disturb_global_rng():
    model = QuantMLP(LadderConfig(g_spec=DY8), sizes=(64, 32, 32, 10))
    x = x_batch(8, 64)
    torch.manual_seed(11)
    before = torch.rand(4)
    torch.manual_seed(11)
    model(x).sum().backward()
    assert torch.equal(before, torch.rand(4))


def test_generators_are_lazy_and_follow_the_parameter_device():
    """torch.Generator cannot be moved across devices, so building them in
    __init__ pins them to CPU; model.to('cuda') would then move parameters but
    not generators and stochastic rounding would raise on its own device check."""
    layer = QuantLinear(8, 4, LadderConfig(a_spec=ACT8), name="fc0", seed=0)
    assert layer._gens == {}, "generators must not be built eagerly"
    layer(x_batch(2, 8))
    assert {k[1] for k in layer._gens} == {"cpu"}


def test_frozen_batches_replay_identically():
    a = frozen_batches(60000, 32, 50, seed=0)
    assert torch.equal(a, frozen_batches(60000, 32, 50, seed=0))
    assert a.shape == (50, 32) and a.min() >= 0 and a.max() < 60000
    assert not torch.equal(a, frozen_batches(60000, 32, 50, seed=1))


# ================================================================= the decoder

def test_attention_is_causal():
    """Changing token t+1 must NOT change the logits at position t. If this
    fails the model peeks at the future and every loss number is invalid."""
    torch.manual_seed(0)
    model = QuantGPT(small_cfg()).eval()
    x = toks()
    with torch.no_grad():
        base, _ = model(x)
        for pos in (5, 10, 14):
            x2 = x.clone()
            x2[:, pos] = (x2[:, pos] + 37) % 128
            alt, _ = model(x2)
            assert torch.allclose(base[:, :pos], alt[:, :pos], atol=1e-6), (
                f"changing token {pos} altered logits before it -- mask leaks")
            assert not torch.allclose(base[:, pos], alt[:, pos])


def test_loss_at_init_is_near_uniform_on_unpredictable_targets():
    """Targets must be INDEPENDENT of inputs. model(x, x) asks the model to
    predict the CURRENT token, which a causal model with tied embeddings can
    partly do (4.46 vs ln(128)=4.85). That is the wrong question, not a leak."""
    torch.manual_seed(0)
    cfg = small_cfg()
    _, loss = QuantGPT(cfg)(toks(seed=0), toks(seed=99))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.15


def test_current_token_visible_but_next_token_is_not():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = QuantGPT(cfg)
    x = toks(seed=0)
    uniform = math.log(cfg.vocab_size)
    assert model(x, x)[1].item() < uniform
    assert abs(model(x, torch.roll(x, -1, 1))[1].item() - uniform) < 0.15


def test_body_dominates_and_a_large_vocab_would_invert_it():
    """The M2 lesson applied to the transformer. Guard the real configuration
    AND document the counterfactual it was chosen to avoid."""
    pb = QuantGPT(GPTConfig(vocab_size=2048, ctx=128, n_layer=6, n_head=6,
                            d_model=192)).param_breakdown()
    assert 1_000_000 <= pb["total"] <= 3_500_000, pb["total"]
    assert pb["body_frac"] > 0.80, pb
    big = QuantGPT(GPTConfig(vocab_size=50257, ctx=128, n_layer=6, n_head=6,
                             d_model=192)).param_breakdown()
    assert big["body_frac"] < 0.30


def test_every_block_linear_is_a_quant_site():
    cfg = small_cfg(LadderConfig(w_spec=TERNARY))
    model = QuantGPT(cfg)
    model(toks(), toks())[1].backward()
    names = {r["site"].rsplit(".", 1)[0] for r in registry.dump_stats()}
    for i in range(cfg.n_layer):
        for part in ("qkv", "attn_proj", "mlp_fc", "mlp_proj"):
            assert f"blk{i}.{part}" in names, f"blk{i}.{part} not quantized"


def test_full_ladder_trains_one_step_without_nan():
    model = QuantGPT(small_cfg(LadderConfig(w_spec=TERNARY, a_spec=ACT8, g_spec=DY8)))
    x = toks()
    _, loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_attention_matmuls_are_not_fused_away():
    """R6 needs QK^T and AV visible. scaled_dot_product_attention would hide
    them in a fused kernel where they cannot be quantized or counted."""
    import inspect

    from models.quant_gpt import CausalSelfAttention

    src = inspect.getsource(CausalSelfAttention.forward)
    assert "scaled_dot_product_attention" not in src
    assert "@" in src


def test_activation_quantization_reaches_the_loss_not_just_the_tensor():
    """Stronger than the tensor-level no-op guard, which proves only that some
    tensor changed -- a rung wired to a dead branch would pass it.

    Measured on LOGITS, not the scalar loss: quantization perturbs the loss in an
    arbitrary DIRECTION, so a single scalar is not monotone in bit-width (4-bit
    moved it 2.38e-5 and 8-bit 2.62e-5, the wrong way round purely by sign). The
    RMS deviation of the logit tensor is the quantity that IS monotone.
    """
    x, y = toks(4, 16, 128, seed=0), toks(4, 16, 128, seed=1)
    logits = {}
    for bits in (None, 8, 4, 2):
        spec = (QuantSpec(kind="int", bits=bits, granularity="row", calib="absmax")
                if bits else QuantSpec(kind="none"))
        torch.manual_seed(0)
        m = QuantGPT(small_cfg(LadderConfig(w_spec=TERNARY, a_spec=spec)), seed=0).eval()
        with torch.no_grad():
            logits[bits] = m(x, y)[0]
    devs = {b: (logits[b] - logits[None]).pow(2).mean().sqrt().item() for b in (8, 4, 2)}
    assert devs[8] > 0, "8-bit activations did not change the logits at all"
    assert devs[4] > devs[8], f"4-bit should deviate more: {devs}"
    assert devs[2] > devs[4], f"2-bit should deviate more: {devs}"
