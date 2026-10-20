"""EmbeddingGemma 2's text tower cut to its first k layers, with its own pooling and head.

EG2's sentence embedding is: embed_tokens (scaled) -> 24 encoder layers, each fed a
per-layer input (PLE) projected from the token embeddings -> final RMSNorm ->
embedding_projection (512 -> 768) per token -> mean over tokens, prompt included -> L2
normalize. A k-layer cut keeps embed_tokens, layers 0..k-1, the first k slices of the PLE
projection, the final norm and the projection, so no new weight is introduced: k = 24 is
EG2 itself. The quantized linears and the token table come from the q4 ONNX that built the index
(dequantized to fp32), so k = 24 reproduces the teacher; norms and scalars, which that file
stores unquantized, come from google/embeddinggemma-2 model.safetensors. The bf16 weights
themselves agree with the q4 teacher on only 0.87 of the top 10.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open
from transformers.models.embedding_gemma2 import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

ROOT = Path(__file__).resolve().parent.parent
PREV = ROOT.parent / "eg2-query-student"
sys.path.insert(0, str(PREV / "run"))
from common import QUERY  # noqa: E402

PT = PREV / "out" / "eg2pt"
Q4 = PREV / "out" / "hf" / "model_q4.onnx"
TGT = PREV / "out" / "targets"
HID = 512


def text_config(k):
    c = json.loads((PT / "config.json").read_text())["text_config"]
    c["num_hidden_layers"] = k
    c["layer_types"] = c["layer_types"][:k]
    c["per_layer_config"] = {i: v for i, v in c["per_layer_config"].items() if int(i) < k}
    types = list(c["layer_types"])
    cfg = EmbeddingGemma2TextConfig(**c)
    # The config forces the last layer to full attention; a cut must keep each layer's own
    # type, or the last layer's RoPE theta changes (1e6 instead of 1e4).
    cfg.layer_types = types
    return cfg


def _nibbles(a):
    a = a.astype(np.uint8)
    return np.stack([a & 15, a >> 4], -1).reshape(*a.shape[:-1], -1)


def dequant(q, scales, zp, rows, cols):
    """4-bit blockwise (block 32 along cols, low nibble first) -> float32 [rows, cols]."""
    nb = cols // 32
    v = _nibbles(q.reshape(rows, -1))[:, :cols].reshape(rows, nb, 32).astype(np.float32)
    z = _nibbles(zp.reshape(rows, -1))[:, :nb].astype(np.float32)
    return ((v - z[..., None]) * scales.reshape(rows, nb)[..., None]).reshape(rows, cols)


def q4_weights(k, rows=None):
    """HF-named fp32 tensors for every quantized weight of layers < k in the q4 ONNX."""
    cfg = json.loads((PT / "config.json").read_text())["text_config"]
    m = onnx.load(str(Q4))
    I = {i.name: i for i in m.graph.initializer}
    sizes = {n.input[1][:-len("_quant")]: {a.name: a.i for a in n.attribute}
             for n in m.graph.node if n.op_type == "MatMulNBits"}

    def W(base, n_rows=None, sel=None):
        get = lambda s: numpy_helper.to_array(I[base + s])  # noqa: E731
        q, sc, zp = get("_quant"), get("_scales"), get("_zp")
        if sel is not None:
            q, sc, zp = q[sel], sc[sel], zp[sel]
        if base in sizes:
            return dequant(q, sc, zp, n_rows or sizes[base]["N"], sizes[base]["K"])
        return dequant(q, sc, zp, len(q), HID)

    # The ONNX table has the sqrt(hidden) embedding scale folded in; the HF module applies it.
    out = {"embed_tokens.weight": W("model_embed_tokens_weight", sel=None if rows is None else np.asarray(rows))
           / np.float32(HID ** 0.5),
           "embedding_projection.weight": W("model_embedding_projection_MatMul_weight"),
           "ple.per_layer_model_projection.weight": W("model_ple_per_layer_model_projection_MatMul_weight")[:k * HID]}
    for i in range(k):
        p, h = f"model_layers_{i}_", f"layers.{i}."
        qkv = W(p + "attn_qkv_proj_MatMul_weight")
        lc = cfg["per_layer_config"].get(f"{i:02d}", {})
        hd = lc.get("head_dim", cfg["head_dim"])
        nq, nkv = cfg["num_attention_heads"] * hd, lc.get("num_key_value_heads", cfg["num_key_value_heads"]) * hd
        out[h + "self_attn.q_proj.weight"] = qkv[:nq]
        out[h + "self_attn.k_proj.weight"] = qkv[nq:nq + nkv]
        out[h + "self_attn.v_proj.weight"] = qkv[nq + nkv:]
        out[h + "self_attn.o_proj.weight"] = W(p + "attn_o_proj_MatMul_weight")
        gu = W(p + "mlp_gate_up_proj_MatMul_weight")
        out[h + "mlp.gate_proj.weight"], out[h + "mlp.up_proj.weight"] = gu[:len(gu) // 2], gu[len(gu) // 2:]
        out[h + "mlp.down_proj.weight"] = W(p + "mlp_down_proj_MatMul_weight")
        out[h + "ple_block.per_layer_input_gate.weight"] = W(p + "ple_per_layer_input_gate_MatMul_weight")
        out[h + "ple_block.per_layer_projection.weight"] = W(p + "ple_per_layer_projection_MatMul_weight")
    return {n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in out.items()}


class Student(torch.nn.Module):
    def __init__(self, k, rows=None, source="q4"):
        """rows: kept vocabulary ids (sorted) for a pruned table; None keeps all 262,144.
        source: "q4" (dequantized teacher weights) or "bf16" (the safetensors release)."""
        super().__init__()
        cfg = text_config(k)
        if rows is not None:
            cfg.vocab_size = len(rows)
        self.k = k
        self.tm = EmbeddingGemma2TextModel(cfg)
        sd = {}
        with safe_open(str(PT / "model.safetensors"), "pt") as f:
            for name in f.keys():
                if not name.startswith("language_model."):
                    continue
                n = name[len("language_model."):]
                if n.startswith("layers.") and int(n.split(".")[1]) >= k:
                    continue
                t = f.get_tensor(name).float()
                if n == "ple.per_layer_model_projection.weight":
                    t = t[:k * HID]
                if n == "embed_tokens.weight" and rows is not None:
                    t = t[torch.as_tensor(rows)]
                sd[n] = t
        if source == "q4":
            q4 = q4_weights(k, rows)
            for n, v in q4.items():
                assert v.shape == sd[n].shape, (n, v.shape, sd[n].shape)
            sd.update(q4)
        missing, unexpected = self.tm.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("embed_scale")]
        assert not missing and not unexpected, (missing, unexpected)

    def forward(self, ids, mask):
        h = self.tm(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(e, dim=-1)


def batches(enc, batch, order=None):
    """Pad token-id lists into (ids, mask) tensors, shortest-first unless an order is given."""
    order = np.argsort([len(e) for e in enc]) if order is None else order
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        S = max(len(enc[r]) for r in rows)
        ids = torch.zeros((len(rows), S), dtype=torch.long)
        mask = torch.zeros((len(rows), S), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, :len(enc[r])] = torch.as_tensor(enc[r])
            mask[b, :len(enc[r])] = 1
        yield rows, ids, mask


def tokenize(T, texts, prompt=QUERY, max_tokens=128, remap=None):
    """Token ids exactly as the teacher builds them; remap = a pruned tokenizer instead."""
    tok = remap or T.tok
    return [tok.encode(prompt + T.clean(t)).ids[:max_tokens] for t in texts]


@torch.no_grad()
def embed(model, enc, batch=128):
    model.eval()
    out = np.empty((len(enc), 768), np.float32)
    for rows, ids, mask in batches(enc, batch):
        out[rows] = model(ids, mask).numpy()
    return out


def load(name):
    return [json.loads(l) for l in (TGT / f"{name}.jsonl").open()], np.load(TGT / f"{name}.eg2.npy")
