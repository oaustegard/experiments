"""The muninn packing recipe ("packing-a-small-model", 2026-11-11) applied to granite-docling-258M as
fake quantization: each weight is rounded to what the packed file would decode to and written back as
float32, so any runtime computes exactly what a packed model computes.

  table   the token table (model.text_model.embed_tokens): 4-bit codes -8..7 in groups of 32 along the
          row, one fp16 scale per group (absmax / 7)
  matrix  every other 2-D Linear weight (vision encoder, connector, decoder): int8 codes, one fp16 scale
          per output row (absmax / 63, codes -63..63, as the gliner/Ettin packs use)
  rest    norms, biases, position embeddings, the patch conv: unchanged (fp32 in the browser)

The decoder's output head is tied to the token table. Arm `pack` uses the 4-bit table for it too (one
copy in the file); arm `pack-head8` stores the head as an int8 matrix (a second copy, +58 MB).

    sizes(state)            bytes each scheme would store
    fake(state, arm)        the rounded state dict
"""
from __future__ import annotations

import torch

TABLE = "embed_tokens.weight"


def _f16(x):
    return x.to(torch.float16).to(torch.float32)


def q4_groups(w, g=32):
    o, k = w.shape
    assert k % g == 0
    x = w.float().reshape(o, k // g, g)
    s = _f16(x.abs().amax(-1, keepdim=True) / 7.0).clamp_min(1e-12)
    return (torch.clamp(torch.round(x / s), -8, 7) * s).reshape(o, k)


def q8_rows(w, r=63):
    x = w.float()
    s = _f16(x.abs().amax(-1, keepdim=True) / r).clamp_min(1e-12)
    return torch.clamp(torch.round(x / s), -r, r) * s


def kind(name, t):
    if name.endswith(TABLE):
        return "table"
    if name.endswith("lm_head.weight"):
        return "head"
    if t.ndim == 2 and name.endswith(".weight") and "embedding" not in name:
        return "matrix"
    return "rest"


def fake(state, arm):
    out = {}
    table = None
    for n, t in state.items():
        k = kind(n, t)
        if k == "table":
            table = q4_groups(t)
            out[n] = table
        elif k == "matrix":
            out[n] = q8_rows(t)
        else:
            out[n] = t.float()
    for n, t in state.items():
        if kind(n, t) == "head":
            out[n] = table.clone() if arm == "pack" else q8_rows(t)
    return out


def sizes(state):
    """Bytes per part under the recipe (codes + fp16 scales; rest as fp32)."""
    b = {"table": 0, "matrix": 0, "rest": 0, "head8": 0}
    for n, t in state.items():
        k = kind(n, t)
        if k == "table":
            b["table"] += t.numel() // 2 + t.numel() // 32 * 2
        elif k == "matrix":
            b["matrix"] += t.numel() + t.shape[0] * 2
        elif k == "head":
            b["head8"] += t.numel() + t.shape[0] * 2
        else:
            b["rest"] += t.numel() * 4
    return b
