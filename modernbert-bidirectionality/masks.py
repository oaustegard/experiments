"""Single source of truth for arm attention masks.

An arm is a per-layer boolean mask `allowed[l, i, j]` (query i may read key j
at layer l), always AND-ed with the model's own base mask (all-true for
full_attention layers, |i-j| <= sliding_window for sliding_attention layers).

Two families:
  - layer axis: some layers are fully "free" (no restriction beyond the base
    mask), the rest are causal (j <= i).
  - position axis: applied identically at EVERY layer as
    causal OR j-in-S, then AND-ed with the base mask.

`allowed(arm, seq_idx, ctx)` is the one entry point; everything else is a
helper it calls. `ctx` carries the per-model, per-sequence data an arm may
need (layer_types, and the punct/masked/attn/rand anchor sets).
"""
from __future__ import annotations

import torch

SEQ_LEN = 256


def causal_tensor(seq_len: int = SEQ_LEN) -> torch.Tensor:
    """allowed[i, j] = True iff j <= i."""
    i_idx = torch.arange(seq_len).unsqueeze(1)
    j_idx = torch.arange(seq_len).unsqueeze(0)
    return j_idx <= i_idx


def base_allowed_tensor(layer_types, sliding_window: int = 64, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """(num_layers, seq_len, seq_len) bool: the model's own per-layer mask."""
    num_layers = len(layer_types)
    i_idx = torch.arange(seq_len).unsqueeze(1)
    j_idx = torch.arange(seq_len).unsqueeze(0)
    local = (i_idx - j_idx).abs() <= sliding_window
    full = torch.ones((seq_len, seq_len), dtype=torch.bool)
    out = torch.empty((num_layers, seq_len, seq_len), dtype=torch.bool)
    for l, t in enumerate(layer_types):
        out[l] = full if t == "full_attention" else local
    return out


def _layer_axis_free_set(arm: str, num_layers: int, layer_types) -> set[int] | None:
    """Returns the set of layers that are fully "free" for a layer-axis arm,
    or None if `arm` is not a layer-axis arm (i.e. it's a position-axis arm)."""
    if arm == "bidir":
        return set(range(num_layers))
    if arm == "causal":
        return set()
    if arm.startswith("single:"):
        l = int(arm.split(":", 1)[1])
        return set(range(num_layers)) - {l}
    if arm.startswith("prefix:"):
        k = int(arm.split(":", 1)[1])
        causal_layers = set(range(0, k))
        return set(range(num_layers)) - causal_layers
    if arm.startswith("suffix:"):
        k = int(arm.split(":", 1)[1])
        causal_layers = set(range(k, num_layers))
        return set(range(num_layers)) - causal_layers
    if arm == "keep:global":
        return {l for l, t in enumerate(layer_types) if t == "full_attention"}
    if arm == "keep:local":
        return {l for l, t in enumerate(layer_types) if t == "sliding_attention"}
    if arm == "keep:every4":
        return set(range(0, num_layers, 4))
    if arm == "keep:first4":
        return set(range(0, 4))
    if arm == "keep:last4":
        return set(range(num_layers - 4, num_layers))
    if arm == "keep:first8":
        return set(range(0, 8))
    if arm == "keep:last8":
        return set(range(num_layers - 8, num_layers))
    if arm in ("keep:top4", "keep:top8"):
        return None  # resolved by caller from ctx["top_layers"], not here
    return None


LAYER_AXIS_PREFIXES = (
    "single:", "prefix:", "suffix:",
)
LAYER_AXIS_EXACT = {
    "bidir", "causal", "keep:global", "keep:local", "keep:every4",
    "keep:first4", "keep:last4", "keep:first8", "keep:last8",
    "keep:top4", "keep:top8",
}


SINK_SUFFIX = "+special"


def strip_sinks(arm: str) -> tuple[str, bool]:
    """'single+special:3' -> ('single:3', True); 'causal+special' -> ('causal', True)."""
    if arm.endswith(SINK_SUFFIX):
        return arm[: -len(SINK_SUFFIX)], True
    head, sep, tail = arm.partition(":")
    if head.endswith(SINK_SUFFIX):
        return head[: -len(SINK_SUFFIX)] + sep + tail, True
    return arm, False


def is_layer_axis_arm(arm: str) -> bool:
    arm, _ = strip_sinks(arm)
    return arm in LAYER_AXIS_EXACT or arm.startswith(LAYER_AXIS_PREFIXES)


def is_seq_dependent_arm(arm: str) -> bool:
    """True iff this arm's mask varies per sequence (needs ctx anchor sets)."""
    if arm in ("anchor:punct", "anchor:masked"):
        return True
    if arm.startswith("anchor:attn:") or arm.startswith("anchor:rand:"):
        return True
    if arm in ("look:8+punct", "look:8+attn:10"):
        return True
    return False


def _position_axis_extra(arm: str, seq_idx: int | None, ctx: dict, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Returns (1, seq_len) or (seq_len, seq_len) bool: allowed beyond causal,
    to be OR-ed with causal and applied identically at every layer."""
    i_idx = torch.arange(seq_len).unsqueeze(1)
    j_idx = torch.arange(seq_len).unsqueeze(0)

    if arm.startswith("look:") and "+" not in arm:
        m = int(arm.split(":", 1)[1])
        return j_idx <= (i_idx + m)

    if arm == "anchor:special":
        col = torch.zeros(seq_len, dtype=torch.bool)
        col[0] = True
        col[-1] = True
        return col.unsqueeze(0)

    if arm == "anchor:first4":
        col = torch.zeros(seq_len, dtype=torch.bool)
        col[:4] = True
        return col.unsqueeze(0)

    if arm == "anchor:punct":
        return ctx["punct"][seq_idx].unsqueeze(0)

    if arm.startswith("anchor:attn:"):
        p = int(arm.split(":")[2])
        return ctx["attn_top"][p][seq_idx].unsqueeze(0)

    if arm.startswith("anchor:rand:"):
        p = int(arm.split(":")[2])
        return ctx["rand"][p][seq_idx].unsqueeze(0)

    if arm == "anchor:masked":
        return ctx["masked"][seq_idx].unsqueeze(0)

    if arm == "look:8+punct":
        near = j_idx <= (i_idx + 8)
        return near | ctx["punct"][seq_idx].unsqueeze(0)

    if arm == "look:8+attn:10":
        near = j_idx <= (i_idx + 8)
        return near | ctx["attn_top"][10][seq_idx].unsqueeze(0)

    raise ValueError(f"unknown arm: {arm!r}")


def allowed(arm: str, seq_idx: int | None, ctx: dict) -> torch.Tensor:
    """(num_layers, seq_len, seq_len) bool allowed mask for one arm.

    ctx must carry: "layer_types" (list[str]), "base" (cached base_allowed
    tensor), "causal" (cached causal tensor), "seq_len", and, for arms that
    need them: "punct", "masked", "attn_top", "rand", "top_layers".
    """
    layer_types = ctx["layer_types"]
    num_layers = len(layer_types)
    seq_len = ctx.get("seq_len", SEQ_LEN)
    base = ctx["base"]
    causal = ctx["causal"]

    arm, sinks = strip_sinks(arm)
    if sinks:
        # a "+special" layer-axis arm: the restricted layers are causal but every
        # query may still read [CLS] (0) and [SEP] (seq_len-1), so the model's
        # sink positions stay reachable. Separates OOD collapse from information loss.
        col = torch.zeros(seq_len, dtype=torch.bool)
        col[0] = True
        col[-1] = True
        causal = causal | col.unsqueeze(0)

    if arm in ("keep:top4", "keep:top8"):
        key = arm.split(":", 1)[1]
        free_layers = set(ctx["top_layers"][key])
    else:
        free_layers = _layer_axis_free_set(arm, num_layers, layer_types)

    out = torch.empty((num_layers, seq_len, seq_len), dtype=torch.bool)

    if free_layers is not None:
        for l in range(num_layers):
            layer_pos = torch.ones((seq_len, seq_len), dtype=torch.bool) if l in free_layers else causal
            out[l] = base[l] & layer_pos
        return out

    # position axis: same extra allowed-set at every layer
    extra = _position_axis_extra(arm, seq_idx, ctx, seq_len)
    layer_pos = causal | extra
    for l in range(num_layers):
        out[l] = base[l] & layer_pos
    return out


def to_additive(allowed_bool: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bool allowed -> additive float mask (0 allowed, finfo(dtype).min blocked)."""
    neg = torch.finfo(dtype).min
    additive = torch.zeros(allowed_bool.shape, dtype=dtype)
    additive.masked_fill_(~allowed_bool, neg)
    return additive


class MaskHolder:
    """Mutable box for the additive mask currently in effect. `.mask` is
    (Bm, num_layers, seq_len, seq_len) float, Bm in {1, batch_size}."""

    __slots__ = ("mask",)

    def __init__(self):
        self.mask = None


def install_mask_hooks(layers):
    """Registers a forward-pre-hook on each ModernBertEncoderLayer that
    replaces its `attention_mask` kwarg with holder.mask[:, layer_idx].

    Returns (holder, handles); call handle.remove() on each handle to undo.
    """
    holder = MaskHolder()

    def make_hook(layer_idx):
        def hook(module, args, kwargs):
            if holder.mask is None:  # pass-through: the stock model's own mask applies
                return None
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = holder.mask[:, layer_idx].unsqueeze(1)
            return args, kwargs
        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(l), with_kwargs=True)
        for l, layer in enumerate(layers)
    ]
    return holder, handles
