"""Checks the LoopCD patch on a tiny random Inkling 2 loop model (CPU or Metal).

    python -m pytest test_loopcd.py -q
"""

import mlx.core as mx
import pytest

import loopcd
from mlx_lm.models.inkling2 import Model, ModelArgs

TEXT = dict(
    hidden_size=64,
    num_hidden_layers=6,
    vocab_size=128,
    unpadded_vocab_size=120,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=16,
    local_layer_ids=[0, 2, 4],
    d_rel=4,
    rel_extent=32,
    log_scaling_n_floor=64,
    log_scaling_alpha=0.1,
    rms_norm_eps=1e-6,
    qk_l2_clip_threshold=22.6,
    rel_l2_clip_threshold=8.0,
    sconv_kernel_size=4,
    n_layers_per_attn_res_block=2,
    n_routed_experts=8,
    num_experts_per_tok=2,
    intermediate_size=32,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    route_scale=1.0,
    final_logit_softcapping=30.0,
    loop_config=dict(core_start=2, core_end=4, num_passes=3, gate_rank=8),
)


@pytest.fixture()
def model():
    mx.random.seed(0)
    m = Model(ModelArgs(model_type="inkling2_loop", text_config=dict(TEXT)))
    # Non-zero depth queries so the block-attention paths are not uniform.
    tm = m.model.language_model
    tm.message_input.query = 0.5 * mx.random.normal(tm.message_input.query.shape)
    mx.eval(m.parameters())
    yield m
    loopcd.disable(m)


TOKENS = mx.array([[5, 17, 3, 99, 42, 7, 64, 11, 2, 80]])


def logits(m, tokens=TOKENS):
    out = m(tokens, cache=m.make_cache())
    mx.eval(out)
    return out


def test_omega_zero_matches_stock(model):
    stock = logits(model)
    loopcd.enable(model, omega=0.0)
    assert mx.allclose(stock, logits(model), atol=1e-5).item()


def test_guidance_changes_output(model):
    stock = logits(model)
    loopcd.enable(model, omega=0.5)
    assert mx.abs(stock - logits(model)).max().item() > 1e-3


def test_hidden_only_differs_from_full(model):
    loopcd.enable(model, omega=0.5)
    full = logits(model)
    loopcd.enable(model, omega=0.5, guide_blocks=False)
    assert mx.abs(full - logits(model)).max().item() > 1e-4


@pytest.mark.parametrize("passes", [3, 2])
def test_prefill_then_decode_matches_one_forward(model, passes):
    """The coda caches hold guided states, so the incremental path must agree
    with a single forward over the whole sequence."""
    loopcd.enable(model, omega=0.5, passes=passes)
    whole = logits(model)[:, -1]
    cache = model.make_cache()
    model(TOKENS[:, :-1], cache=cache)
    step = model(TOKENS[:, -1:], cache=cache)[:, -1]
    mx.eval(whole, step)
    assert mx.allclose(whole, step, atol=1e-4).item()


def test_passes_resizes_cache_and_restores(model):
    stock_len = len(model.make_cache())  # 2 + 3*2 + 2
    loopcd.enable(model, omega=0.3, passes=2)
    assert len(model.make_cache()) == stock_len - 2
    logits(model)
    loopcd.disable(model)
    assert len(model.make_cache()) == stock_len


def test_unguided_prompt_only_changes_prefill(model):
    loopcd.enable(model, omega=0.5, guide_prompt=False)
    a = logits(model)
    loopcd.disable(model)
    assert mx.allclose(a, logits(model), atol=1e-5).item()
