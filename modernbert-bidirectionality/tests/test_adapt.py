import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adapt  # noqa: E402
import masks  # noqa: E402
import probe  # noqa: E402

SEQ_LEN = 256
# matches jhu-clsp/ettin-decoder-32m: full at 0,3,6,9, sliding elsewhere
LAYER_TYPES = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(10)]
SLIDING_WINDOW = 64


def _blocked(additive: torch.Tensor, l: int, i: int, j: int) -> bool:
    return additive[0, l, i, j].item() < -1.0


def _allowed(additive: torch.Tensor, l: int, i: int, j: int) -> bool:
    return additive[0, l, i, j].item() == 0.0


# ---------------------------------------------------------------------------
# (a) arm masks

def test_causal_blocks_future_at_every_layer():
    m = adapt.build_arm_mask("causal", LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    i = 100
    for l in range(len(LAYER_TYPES)):
        assert _blocked(m, l, i, i + 1)
        assert _allowed(m, l, i, i)


def test_look8_boundary_at_every_layer():
    m = adapt.build_arm_mask("look8", LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    i = 100
    for l in range(len(LAYER_TYPES)):
        assert _allowed(m, l, i, i + 8)
        assert _blocked(m, l, i, i + 9)


def test_global_open_at_full_layers_causal_elsewhere():
    m = adapt.build_arm_mask("global", LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    base = masks.base_allowed_tensor(LAYER_TYPES, sliding_window=SLIDING_WINDOW, seq_len=SEQ_LEN)
    i, j = 50, 150  # j > i, within sliding window distance (100 <= 64? no -> also tests window)
    i2, j2 = 100, 130  # distance 30, within sliding window, future position
    for l, t in enumerate(LAYER_TYPES):
        if t == "full_attention":
            # fully open within the base mask everywhere
            for ii in (0, 50, 200):
                for jj in (0, 128, 255):
                    assert _allowed(m, l, ii, jj) == bool(base[l, ii, jj])
        else:
            # causal: future blocked even when within the sliding window
            assert _blocked(m, l, i2, j2)
            assert _allowed(m, l, i2, i2)


def test_both_open_full_look8_sliding():
    m = adapt.build_arm_mask("both", LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    base = masks.base_allowed_tensor(LAYER_TYPES, sliding_window=SLIDING_WINDOW, seq_len=SEQ_LEN)
    i = 100
    for l, t in enumerate(LAYER_TYPES):
        if t == "full_attention":
            for jj in (0, 128, 255):
                assert _allowed(m, l, i, jj) == bool(base[l, i, jj])
        else:
            assert _allowed(m, l, i, i + 8)
            assert _blocked(m, l, i, i + 9)


def test_bidir_equals_base():
    m = adapt.build_arm_mask("bidir", LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    base = masks.base_allowed_tensor(LAYER_TYPES, sliding_window=SLIDING_WINDOW, seq_len=SEQ_LEN)
    expected = masks.to_additive(base, torch.float32).unsqueeze(0)
    assert torch.equal(m, expected)


@pytest.mark.parametrize("arm", list(adapt.ARMS))
def test_every_arm_blocks_beyond_sliding_window(arm):
    m = adapt.build_arm_mask(arm, LAYER_TYPES, SLIDING_WINDOW, torch.float32)
    sliding_layers = [l for l, t in enumerate(LAYER_TYPES) if t == "sliding_attention"]
    i, j = 100, 100 + SLIDING_WINDOW + 6  # |i-j| = 70 > 64
    for l in sliding_layers:
        assert _blocked(m, l, i, j)


# ---------------------------------------------------------------------------
# (b) mask_batch

def _sample_ids(n=5, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(1000, 40000, size=(n, SEQ_LEN)).astype(np.int32)


def test_mask_batch_deterministic():
    ids = _sample_ids()
    m1, l1, b1 = adapt.mask_batch(ids, step=3)
    m2, l2, b2 = adapt.mask_batch(ids, step=3)
    assert np.array_equal(m1, m2)
    assert np.array_equal(l1, l2)
    assert np.array_equal(b1, b2)


def test_mask_batch_different_step_differs():
    ids = _sample_ids()
    _, _, b1 = adapt.mask_batch(ids, step=3)
    _, _, b2 = adapt.mask_batch(ids, step=4)
    assert not np.array_equal(b1, b2)


def test_mask_batch_never_masks_ends():
    ids = _sample_ids()
    _, _, masked_bool = adapt.mask_batch(ids, step=7)
    assert not masked_bool[:, 0].any()
    assert not masked_bool[:, SEQ_LEN - 1].any()


def test_mask_batch_rate_and_labels():
    ids = _sample_ids(n=10)
    masked_ids, labels, masked_bool = adapt.mask_batch(ids, step=11)
    counts = masked_bool.sum(axis=1)
    target = round(254 * 0.15)
    assert np.all(np.abs(counts - target) <= 1)

    assert np.array_equal(labels != -100, masked_bool)
    assert np.all(labels[masked_bool] == ids[masked_bool])
    assert np.all(masked_ids[masked_bool] == probe.EXPECTED_SPECIAL_IDS["mask"])
    assert np.array_equal(masked_ids[~masked_bool], ids[~masked_bool])


# ---------------------------------------------------------------------------
# (c), (d) real model

@pytest.fixture(scope="module")
def loaded_model():
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(adapt.MODEL_NAME, attn_implementation="sdpa")
    model.eval()
    layer_types = list(model.config.layer_types)
    sliding_window = model.config.sliding_window
    holder, handles = masks.install_mask_hooks(model.model.layers)
    yield model, holder, layer_types, sliding_window
    for h in handles:
        h.remove()


def test_causal_arm_is_position_invariant_to_future_tokens(loaded_model):
    model, holder, layer_types, sliding_window = loaded_model
    dtype = next(model.parameters()).dtype
    causal_mask = adapt.build_arm_mask("causal", layer_types, sliding_window, dtype)

    torch.manual_seed(0)
    x = torch.randint(1000, 40000, (1, SEQ_LEN))
    i = 100
    sel = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    sel[0, i] = True

    holder.mask = causal_mask
    with torch.no_grad():
        logits_before = adapt.masked_logits(model, x, sel)

    x_perturbed = x.clone()
    x_perturbed[0, i + 1 :] = torch.randint(1000, 40000, (SEQ_LEN - i - 1,))

    holder.mask = causal_mask
    with torch.no_grad():
        logits_after = adapt.masked_logits(model, x_perturbed, sel)

    assert torch.allclose(logits_before, logits_after, atol=1e-4)


def test_bidir_arm_is_sensitive_to_future_tokens(loaded_model):
    model, holder, layer_types, sliding_window = loaded_model
    dtype = next(model.parameters()).dtype
    bidir_mask = adapt.build_arm_mask("bidir", layer_types, sliding_window, dtype)

    torch.manual_seed(1)
    x = torch.randint(1000, 40000, (1, SEQ_LEN))
    i = 100
    sel = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    sel[0, i] = True

    holder.mask = bidir_mask
    with torch.no_grad():
        logits_before = adapt.masked_logits(model, x, sel)

    x_perturbed = x.clone()
    x_perturbed[0, i + 1 :] = torch.randint(1000, 40000, (SEQ_LEN - i - 1,))

    holder.mask = bidir_mask
    with torch.no_grad():
        logits_after = adapt.masked_logits(model, x_perturbed, sel)

    assert not torch.allclose(logits_before, logits_after, atol=1e-4)


def test_evaluate_returns_sane_metrics(loaded_model):
    model, holder, layer_types, sliding_window = loaded_model
    dtype = next(model.parameters()).dtype
    causal_mask = adapt.build_arm_mask("causal", layer_types, sliding_window, dtype)

    ids = probe.load_corpus_ids(256)[:8]
    result = adapt.evaluate(model, holder, causal_mask, ids, batch_size=8)

    assert result is not None
    assert np.isfinite(result["ce"])
    assert result["ce"] > 0
    assert 0.0 <= result["acc"] <= 1.0
