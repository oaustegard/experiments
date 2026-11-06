"""Ettin-17m (ModernBERT) embedder in plain PyTorch, mirroring web/distill.js exactly.

Embedder = token table -> LayerNorm (no bias) -> 7 layers (pre-LN, RoPE theta 160000 on every layer,
local layers see |i-j| <= 64) -> final LayerNorm -> mean pool over real tokens -> Dense 256->768
(no bias). Flat parameter layout (layout()) is the one the WGSL trainer uses:
table, emb_norm, per layer [attn_norm (absent in layer 0), Wqkv, Wo, mlp_norm, Wi, Wo_mlp], final_norm, dense.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

H, NH, D, I, L, WIN, THETA, EPS = 256, 4, 64, 384, 7, 64, 160000.0, 1e-5
GLOBAL = [i % 3 == 0 for i in range(L)]
QP = "Represent this sentence for searching relevant passages: "


def names(V):
    out = [("table", (V, H)), ("emb_norm", (H,))]
    for i in range(L):
        if i > 0:
            out.append((f"L{i}.attn_norm", (H,)))
        out += [(f"L{i}.Wqkv", (3 * H, H)), (f"L{i}.Wo", (H, H)), (f"L{i}.mlp_norm", (H,)),
                (f"L{i}.Wi", (2 * I, H)), (f"L{i}.Wo_mlp", (H, I))]
    return out + [("final_norm", (H,)), ("dense", (768, H))]


def layout(V):
    off, out = 0, []
    for n, s in names(V):
        out.append({"name": n, "shape": list(s), "offset": off})
        off += int(np.prod(s))
    return out, off


HF = {"table": "embeddings.tok_embeddings.weight", "emb_norm": "embeddings.norm.weight", "final_norm": "final_norm.weight"}
for _i in range(L):
    HF |= {f"L{_i}.attn_norm": f"layers.{_i}.attn_norm.weight", f"L{_i}.Wqkv": f"layers.{_i}.attn.Wqkv.weight",
           f"L{_i}.Wo": f"layers.{_i}.attn.Wo.weight", f"L{_i}.mlp_norm": f"layers.{_i}.mlp_norm.weight",
           f"L{_i}.Wi": f"layers.{_i}.mlp.Wi.weight", f"L{_i}.Wo_mlp": f"layers.{_i}.mlp.Wo.weight"}


class Embedder(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace(".", "_"): torch.nn.Parameter(v.clone().float()) for k, v in params.items()})
        inv = 1.0 / (THETA ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        self.register_buffer("inv", inv, persistent=False)

    def P(self, n):
        return self.p[n.replace(".", "_")]

    def ordered(self, V=None):
        V = V or self.P("table").shape[0]
        return [(n, self.P(n)) for n, _ in names(V)]

    def rope(self, x):  # x [B, nh, S, D]
        S = x.shape[2]
        a = torch.arange(S, dtype=torch.float32)[:, None] * self.inv[None]
        c, s = torch.cos(a), torch.sin(a)
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def hidden(self, ids, att):
        B, S = ids.shape
        x = F.layer_norm(F.embedding(ids, self.P("table")), (H,), self.P("emb_norm"), None, EPS)
        key = att.bool()[:, None, None, :]
        pos = torch.arange(S)
        local = ((pos[:, None] - pos[None]).abs() <= WIN)[None, None]
        for i in range(L):
            h = x if i == 0 else F.layer_norm(x, (H,), self.P(f"L{i}.attn_norm"), None, EPS)
            q, k, v = (h @ self.P(f"L{i}.Wqkv").T).view(B, S, 3, NH, D).permute(2, 0, 3, 1, 4)
            q, k = self.rope(q), self.rope(k)
            mask = key if GLOBAL[i] else key & local
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + o.transpose(1, 2).reshape(B, S, H) @ self.P(f"L{i}.Wo").T
            h = F.layer_norm(x, (H,), self.P(f"L{i}.mlp_norm"), None, EPS)
            u = h @ self.P(f"L{i}.Wi").T
            x = x + (F.gelu(u[..., :I]) * u[..., I:]) @ self.P(f"L{i}.Wo_mlp").T
        return F.layer_norm(x, (H,), self.P("final_norm"), None, EPS)

    def forward(self, ids, att):
        hs = self.hidden(ids, att)
        m = att.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1)
        return pooled @ self.P("dense").T


def load_st(path):
    """params dict from a SentenceTransformer dir (e17c)."""
    path = Path(path)
    sd = load_file(str(path / "model.safetensors"))
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    p = {n: sd[HF[n]] for n in HF if HF[n] in sd}
    p["dense"] = load_file(str(path / "2_Dense" / "model.safetensors"))["linear.weight"]
    return p


def to_flat(m: Embedder):
    return torch.cat([t.detach().reshape(-1) for _, t in m.ordered()]).numpy().astype(np.float32)


def from_flat(flat, V):
    lay, n = layout(V)
    assert len(flat) == n, (len(flat), n)
    return {e["name"]: torch.from_numpy(np.asarray(flat[e["offset"]: e["offset"] + int(np.prod(e["shape"]))], np.float32).reshape(e["shape"]).copy()) for e in lay}
