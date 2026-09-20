import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import masks as M  # noqa: E402
import probe  # noqa: E402

MODEL_NAME = "answerdotai/ModernBERT-base"
SEQ_LEN = 256
LAYER_TYPES = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(22)]


def base_ctx(masked_bool=None):
    base = M.base_allowed_tensor(LAYER_TYPES, sliding_window=64, seq_len=SEQ_LEN)
    causal = M.causal_tensor(seq_len=SEQ_LEN)
    ctx = {"layer_types": LAYER_TYPES, "seq_len": SEQ_LEN, "base": base, "causal": causal}
    if masked_bool is not None:
        ctx["masked"] = [torch.from_numpy(masked_bool)]
    return ctx


def _model_ctx(model):
    layer_types = list(model.config.layer_types)
    return {
        "layer_types": layer_types,
        "seq_len": SEQ_LEN,
        "base": M.base_allowed_tensor(layer_types, sliding_window=model.config.sliding_window, seq_len=SEQ_LEN),
        "causal": M.causal_tensor(seq_len=SEQ_LEN),
    }


# ---------------------------------------------------------------------------
# (a) all-true (bidir) arm reproduces stock logits

def test_bidir_reproduces_stock_logits():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(1000, 5000, (2, SEQ_LEN))
    input_ids[:, 0] = tok.cls_token_id
    input_ids[:, -1] = tok.sep_token_id
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        stock_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

    ctx = _model_ctx(model)
    allowed_bool = M.allowed("bidir", None, ctx)
    additive = M.to_additive(allowed_bool, next(model.parameters()).dtype).unsqueeze(0)

    holder, handles = M.install_mask_hooks(model.model.layers)
    try:
        holder.mask = additive
        with torch.no_grad():
            hooked_logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    finally:
        for h in handles:
            h.remove()

    assert torch.allclose(stock_logits, hooked_logits, atol=1e-4)


# ---------------------------------------------------------------------------
# (b) causal blocks future influence

def test_causal_blocks_future_influence():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    torch.manual_seed(1)
    input_ids = torch.randint(1000, 5000, (1, SEQ_LEN))
    input_ids[:, 0] = tok.cls_token_id
    input_ids[:, -1] = tok.sep_token_id

    ctx = _model_ctx(model)
    allowed_bool = M.allowed("causal", None, ctx)
    additive = M.to_additive(allowed_bool, next(model.parameters()).dtype).unsqueeze(0)

    i = 50
    perturbed = input_ids.clone()
    perturbed[:, i + 1:] = torch.randint(1000, 5000, (1, SEQ_LEN - i - 1))

    holder, handles = M.install_mask_hooks(model.model.layers)
    try:
        holder.mask = additive
        with torch.no_grad():
            out1 = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
            out2 = model(input_ids=perturbed, attention_mask=torch.ones_like(perturbed)).logits
    finally:
        for h in handles:
            h.remove()

    assert torch.allclose(out1[:, :i + 1], out2[:, :i + 1], atol=1e-4)
    assert not torch.allclose(out1[:, i + 1:], out2[:, i + 1:], atol=1e-4)


# ---------------------------------------------------------------------------
# (c) mask-shape unit tests, no model

def test_sliding_window_enforced_in_every_arm():
    ctx = base_ctx()
    arms = ["bidir", "causal", "single:5", "prefix:3", "suffix:10", "look:8", "anchor:special"]
    sliding_layer = 1
    assert LAYER_TYPES[sliding_layer] == "sliding_attention"
    i_idx = torch.arange(SEQ_LEN).unsqueeze(1)
    j_idx = torch.arange(SEQ_LEN).unsqueeze(0)
    far = (i_idx - j_idx).abs() > 64
    for arm in arms:
        seq_idx = 0 if M.is_seq_dependent_arm(arm) else None
        a = M.allowed(arm, seq_idx, ctx)
        assert not a[sliding_layer][far].any(), arm


def test_single_layer_differs_only_at_that_layer():
    ctx = base_ctx()
    bidir = M.allowed("bidir", None, ctx)
    single5 = M.allowed("single:5", None, ctx)
    for l in range(22):
        same = torch.equal(bidir[l], single5[l])
        assert same == (l != 5), l


def test_prefix_causal_exactly_at_named_layers():
    ctx = base_ctx()
    causal = M.allowed("causal", None, ctx)
    prefix3 = M.allowed("prefix:3", None, ctx)
    for l in range(22):
        expect_causal = l in (0, 1, 2)
        assert torch.equal(prefix3[l], causal[l]) == expect_causal, l


def test_suffix_causal_exactly_at_named_layers():
    ctx = base_ctx()
    causal = M.allowed("causal", None, ctx)
    suffix20 = M.allowed("suffix:20", None, ctx)
    for l in range(22):
        expect_causal = l in (20, 21)
        assert torch.equal(suffix20[l], causal[l]) == expect_causal, l


def test_keep_global_bidirectional_exactly_at_full_layers():
    ctx = base_ctx()
    bidir = M.allowed("bidir", None, ctx)
    causal = M.allowed("causal", None, ctx)
    kg = M.allowed("keep:global", None, ctx)
    for l, t in enumerate(LAYER_TYPES):
        if t == "full_attention":
            assert torch.equal(kg[l], bidir[l])
        else:
            assert torch.equal(kg[l], causal[l])


def test_look_m_window():
    ctx = base_ctx()
    look4 = M.allowed("look:4", None, ctx)
    full_layer = 0
    assert LAYER_TYPES[full_layer] == "full_attention"
    i = 100
    for j in range(0, i + 6):
        expected = j <= i + 4
        assert bool(look4[full_layer, i, j]) == expected, (i, j)


def test_anchor_special():
    ctx = base_ctx()
    a = M.allowed("anchor:special", None, ctx)
    full_layer = 0
    assert a[full_layer, :, 0].all()
    assert a[full_layer, :, SEQ_LEN - 1].all()
    i, j = 10, 11  # non-special, future -> blocked
    assert not a[full_layer, i, j]


def test_anchor_masked():
    masked_bool = np.zeros(SEQ_LEN, dtype=bool)
    masked_bool[[20, 200]] = True
    ctx = base_ctx(masked_bool=masked_bool)
    a = M.allowed("anchor:masked", 0, ctx)
    full_layer = 0
    i = 5
    assert a[full_layer, i, 200]  # masked, future -> allowed
    assert not a[full_layer, i, 199]  # unmasked, future -> blocked


# ---------------------------------------------------------------------------
# (d) masking draw is deterministic and avoids position 0 / seq_len-1

def test_masking_draw_deterministic_and_avoids_specials():
    corpus_ids = np.tile(np.arange(SEQ_LEN, dtype=np.int64), (4, 1))
    m1, l1, b1 = probe.build_masked_inputs(corpus_ids)
    m2, l2, b2 = probe.build_masked_inputs(corpus_ids)
    assert np.array_equal(m1, m2)
    assert np.array_equal(l1, l2)
    assert np.array_equal(b1, b2)
    assert not b1[:, 0].any()
    assert not b1[:, -1].any()


# --- additions after the smoke: sink-reachable variants, anchor:first4, hook pass-through

def _ctx22():
    import masks as M
    lt = ["full_attention" if l % 3 == 0 else "sliding_attention" for l in range(22)]
    return {"layer_types": lt, "base": M.base_allowed_tensor(lt), "causal": M.causal_tensor(), "seq_len": 256}


def test_anchor_first4_allows_positions_0_to_3_only():
    import masks as M
    ctx = _ctx22()
    a = M.allowed("anchor:first4", None, ctx)
    for l in range(22):
        row = 100 if ctx["layer_types"][l] == "full_attention" else 30  # sliding layers only reach 64 back
        assert bool(a[l, row, 0]) and bool(a[l, row, 3])
        assert not bool(a[l, row, row + 1])
        assert bool(a[l, row, row])
    assert not bool(a[1, 100, 0])  # a sliding layer cannot reach position 0 from row 100 in any arm


def test_plus_special_keeps_sinks_reachable_in_causal_layers():
    import masks as M
    ctx = _ctx22()
    plain = M.allowed("single:5", None, ctx)
    sink = M.allowed("single+special:5", None, ctx)
    # layer 5 (sliding): causal, but column 255 is outside the window at row 10; column 0 is inside
    assert bool(plain[5, 10, 0]) and bool(sink[5, 10, 0])          # j=0 <= i: causal already allows it
    assert not bool(plain[5, 10, 11]) and not bool(sink[5, 10, 11])  # a non-sink future key stays blocked
    assert bool(plain[6, 10, 255]) and bool(sink[6, 10, 255])
    # layer 0 (global) is free in both
    assert torch.equal(plain[0], sink[0])
    full = M.allowed("causal+special", None, ctx)
    assert bool(full[0, 10, 255]) and not bool(full[0, 10, 200])
    assert M.strip_sinks("single+special:5") == ("single:5", True)
    assert M.strip_sinks("keep:global+special") == ("keep:global", True)
    assert M.strip_sinks("look:8") == ("look:8", False)


def test_hook_passthrough_when_mask_is_none():
    import masks as M
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager").eval()
    holder, handles = M.install_mask_hooks(_MODEL.model.layers)
    try:
        holder.mask = None
        enc = _TOK(["a short sentence about lobsters ."], return_tensors="pt")
        with torch.no_grad():
            hooked = _MODEL(**enc).logits
        for h in handles:
            h.remove()
        handles = []
        with torch.no_grad():
            stock = _MODEL(**enc).logits
        assert torch.allclose(hooked, stock, atol=1e-5)
    finally:
        for h in handles:
            h.remove()
