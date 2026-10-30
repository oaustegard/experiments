"""LFM3 forward pass in plain PyTorch, written from LiquidAI's modeling_lfm3.py and the onnx-community
q4 graph rather than imported (the release's modeling code is remote code).

Weights come from either source, as a flat dict of float tensors keyed like the bf16 release:
  bf16(path)            LiquidAI/LFM3-1.7B model.safetensors
  onnx_q4(model_dir)    onnx-community/LFM3-1.7B-ONNX model_q4.onnx(_data), dequantized

Per layer (sandwich norms): h = op(norm(x)); x += post_op_norm(h); x += post_ffn_norm(mlp(ffn_norm(x))).
  conv: in_proj -> B, C, x; y = C * causal depthwise conv7(B * x); out_proj
  attention: q/k/v/gate projections, RMS norm per 128-wide head on q and k, RoPE (sliding layers:
  theta 1e4 on all 64 pairs; full layers: theta 1e6 on the first 32 pairs only), grouped-query
  attention (16 query heads, 4 KV heads; sliding layers see the last 256 positions), each head's
  output times sigmoid(gate), o_proj.
  mlp: w2(silu(w1 x) * w3 x)
Final norm, then the tied token table as lm_head.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
M3 = Path(__import__("os").environ.get("LFM3_DIR", ROOT / "model3"))
DEV = __import__("os").environ.get("DEV", "cpu")


def config(d=M3):
    return json.loads((Path(d) / "config.json").read_text())


def linear_names(cfg):
    """Every weight matrix the quantizer touches, in forward order, with the name of its input."""
    out = []
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}."
        if kind == "conv":
            out += [(p + "conv.in_proj.weight", f"{i}.op_in"), (p + "conv.out_proj.weight", f"{i}.conv_out")]
        else:
            out += [(p + f"self_attn.{n}.weight", f"{i}.op_in") for n in ("q_proj", "k_proj", "v_proj", "gate_proj")]
            out += [(p + "self_attn.o_proj.weight", f"{i}.attn_out")]
        out += [(p + "feed_forward.w1.weight", f"{i}.ffn_in"), (p + "feed_forward.w3.weight", f"{i}.ffn_in"),
                (p + "feed_forward.w2.weight", f"{i}.ffn_mid")]
    return out


def bf16(path=M3 / "bf16" / "model.safetensors"):
    from safetensors.torch import load_file
    return load_file(str(path))


def onnx_q4(d=M3):
    """Dequantize every MatMulNBits / GatherBlockQuantized weight of model_q4.onnx into release names."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(Path(d) / "model_q4.onnx"), load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    raw = np.memmap(Path(d) / "model_q4.onnx_data", dtype=np.uint8, mode="r")

    def arr(name):
        t = inits[name]
        if not t.external_data:
            return numpy_helper.to_array(t)
        ed = {e.key: e.value for e in t.external_data}
        off, ln = int(ed["offset"]), int(ed["length"])
        dt = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
        return np.frombuffer(raw[off:off + ln].tobytes(), dtype=dt).reshape(tuple(t.dims))

    def deq(stem, N, K, zp=True):
        q = arr(stem + "_quant").reshape(N, -1)                      # [N, K/2] bytes, low nibble first
        nib = np.stack([q & 15, q >> 4], -1).reshape(N, K).astype(np.float32)
        s = arr(stem + "_scales").astype(np.float32).reshape(N, K // 32)
        if zp and (stem + "_zp") in inits:
            z = arr(stem + "_zp").reshape(N, -1)
            z = np.stack([z & 15, z >> 4], -1).reshape(N, -1)[:, :K // 32].astype(np.float32)
        else:
            z = np.full((N, K // 32), 8.0, np.float32)
        return torch.from_numpy(((nib.reshape(N, K // 32, 32) - z[..., None]) * s[..., None]).reshape(N, K))

    nodes = {n.input[1]: n for n in m.graph.node if n.op_type == "MatMulNBits"}

    def mm(stem):
        a = {x.name: onnx.helper.get_attribute_value(x) for x in nodes[stem + "_quant"].attribute}
        return deq(stem, a["N"], a["K"], zp=len(nodes[stem + "_quant"].input) > 3 and nodes[stem + "_quant"].input[3] != "")

    cfg = config(d)
    W = {}
    W["model.embed_tokens.weight"] = deq("model_embed_tokens_weight", cfg["vocab_size"], cfg["hidden_size"])
    for i, kind in enumerate(cfg["layer_types"]):
        p, s = f"model.layers.{i}.", f"model_layers_{i}_"
        for n in ("operator_layernorm", "ffn_layernorm", "post_operator_layernorm", "post_ffn_layernorm"):
            rel = {"operator_layernorm": "operator_norm", "ffn_layernorm": "ffn_norm",
                   "post_operator_layernorm": "post_operator_norm", "post_ffn_layernorm": "post_ffn_norm"}[n]
            W[p + rel + ".weight"] = torch.from_numpy(arr(f"model.layers.{i}.{n}.weight").astype(np.float32).copy())
        if kind == "conv":
            W[p + "conv.in_proj.weight"] = mm(s + "conv_in_proj_MatMul_weight")
            W[p + "conv.out_proj.weight"] = mm(s + "conv_out_proj_MatMul_weight")
            W[p + "conv.conv.weight"] = torch.from_numpy(arr(f"model.layers.{i}.conv.conv.weight").astype(np.float32).copy())
        else:
            qg = mm(s + "attn_q_proj_MatMul_weight")                   # q (2048) and the 16 gate rows fused
            W[p + "self_attn.q_proj.weight"], W[p + "self_attn.gate_proj.weight"] = qg[:-16], qg[-16:]
            W[p + "self_attn.k_proj.weight"] = mm(s + "attn_k_proj_MatMul_weight")
            W[p + "self_attn.v_proj.weight"] = mm(s + "attn_v_proj_MatMul_weight")
            W[p + "self_attn.o_proj.weight"] = mm(s + "attn_o_proj_MatMul_weight")
            for n in ("q_norm", "k_norm"):
                W[p + f"self_attn.{n}.weight"] = torch.from_numpy(
                    arr(f"model.layers.{i}.attn.{n}.layernorm.weight").astype(np.float32).copy())
        W[p + "feed_forward.w1.weight"] = mm(s + "mlp_gate_proj_MatMul_weight")
        W[p + "feed_forward.w3.weight"] = mm(s + "mlp_up_proj_MatMul_weight")
        W[p + "feed_forward.w2.weight"] = mm(s + "mlp_down_proj_MatMul_weight")
    W["model.embedding_norm.weight"] = torch.from_numpy(
        arr(f"model.layers.{len(cfg['layer_types'])}.final_norm_layernorm.weight").astype(np.float32).copy())
    return W


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(cfg, kind, T, offset=0):
    """cos, sin [T, head_dim / 2] of the rotated pairs: `angles` frequencies, the first of which follow
    the full head dimension's progression (partial RoPE keeps that progression)."""
    hd = cfg["head_dim"]
    rp = cfg["rope_parameters"][kind]
    angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
    inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.float) / hd))
    f = torch.arange(offset, offset + T, dtype=torch.float, device=DEV)[:, None] * inv.to(DEV)
    return f.cos(), f.sin()


def rope(x, cos, sin, ort):
    """Rotate pairs of x [T, heads, hd]. Release layout: pair (i, i + hd/2) for i < angles, the rest
    unrotated. onnxruntime layout (the ONNX export permutes q/k head dims to it): pair (i, i + angles)
    inside the first 2 * angles dims. The two are the same attention up to that permutation."""
    hd, n = x.shape[-1], cos.shape[-1]
    c, s = cos[:, None], sin[:, None]
    a, b = (x[..., :n], x[..., n:2 * n]) if ort else (x[..., :n], x[..., hd // 2:hd // 2 + n])
    ra, rb = a * c - b * s, b * c + a * s
    out = x.clone()
    if ort:
        out[..., :n], out[..., n:2 * n] = ra, rb
    else:
        out[..., :n], out[..., hd // 2:hd // 2 + n] = ra, rb
    return out


class Lfm3:
    """Full-sequence forward (no cache), float32. `W` may hold any float dtype; matrices are cast per use.
    `hook(name, x)` sees the input of every quantizable matrix (see linear_names)."""

    def __init__(self, W, cfg=None, ort=False):
        """ort: weights from the ONNX export, whose q/k head dims are in onnxruntime's RoPE layout."""
        self.W, self.cfg, self.ort = W, cfg or config(), ort
        self.hook = None

    def w(self, name):
        return self.W[name].to(DEV).float()

    def lin(self, x, name, tag):
        if self.hook:
            self.hook(tag, x)
        return x @ self.w(name).T

    def layer(self, i, x):
        c, p = self.cfg, f"model.layers.{i}."
        eps, T = c["norm_eps"], x.shape[0]
        kind = c["layer_types"][i]
        h = rms(x, self.w(p + "operator_norm.weight"), eps)
        if kind == "conv":
            bcx = self.lin(h, p + "conv.in_proj.weight", f"{i}.op_in")
            B, C, xx = bcx.chunk(3, -1)
            k = self.w(p + "conv.conv.weight")                       # [D, 1, L]
            y = F.conv1d((B * xx).T[None], k, padding=k.shape[-1] - 1, groups=k.shape[0])[0, :, :T].T
            o = self.lin(C * y, p + "conv.out_proj.weight", f"{i}.conv_out")
        else:
            nh, nkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            q = self.lin(h, p + "self_attn.q_proj.weight", f"{i}.op_in").view(T, nh, hd)
            k = self.lin(h, p + "self_attn.k_proj.weight", f"{i}.op_in").view(T, nkv, hd)
            v = self.lin(h, p + "self_attn.v_proj.weight", f"{i}.op_in").view(T, nkv, hd)
            g = torch.sigmoid(self.lin(h, p + "self_attn.gate_proj.weight", f"{i}.op_in"))
            q = rms(q, self.w(p + "self_attn.q_norm.weight"), eps)
            k = rms(k, self.w(p + "self_attn.k_norm.weight"), eps)
            cos, sin = rope_tables(c, kind, T)
            q, k = rope(q, cos, sin, self.ort), rope(k, cos, sin, self.ort)
            k = k.repeat_interleave(nh // nkv, 1); v = v.repeat_interleave(nh // nkv, 1)
            pos = torch.arange(T, device=DEV)
            allow = pos[:, None] >= pos[None, :]
            if kind == "sliding_attention":
                allow &= (pos[:, None] - pos[None, :]) < c["sliding_window"]
            s = torch.einsum("thd,shd->hts", q, k) * hd ** -0.5
            s = s.masked_fill(~allow, float("-inf")).softmax(-1)
            a = torch.einsum("hts,shd->thd", s, v) * g[..., None]
            o = self.lin(a.reshape(T, nh * hd), p + "self_attn.o_proj.weight", f"{i}.attn_out")
        x = x + rms(o, self.w(p + "post_operator_norm.weight"), eps)
        h = rms(x, self.w(p + "ffn_norm.weight"), eps)
        m = F.silu(self.lin(h, p + "feed_forward.w1.weight", f"{i}.ffn_in")) * self.lin(h, p + "feed_forward.w3.weight", f"{i}.ffn_in")
        x = x + rms(self.lin(m, p + "feed_forward.w2.weight", f"{i}.ffn_mid"), self.w(p + "post_ffn_norm.weight"), eps)
        return x

    @torch.no_grad()
    def hidden(self, ids):
        x = self.w("model.embed_tokens.weight")[torch.as_tensor(ids, device=DEV)]
        for i in range(len(self.cfg["layer_types"])):
            x = self.layer(i, x)
        return rms(x, self.w("model.embedding_norm.weight"), self.cfg["norm_eps"])

    @torch.no_grad()
    def logits(self, ids, last=None):
        h = self.hidden(ids)
        if last:
            h = h[-last:]
        return h @ self.w("model.embed_tokens.weight").T
