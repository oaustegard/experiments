"""Write web/manifest.json for the WGSL LFM2 engine from onnx-community's model_q4.onnx.

  python3 lfm-wgsl/run/manifest.py [MODEL_DIR]      (default lfm-wgsl/model)

The engine keeps one GPU buffer per operator: a q4 matmul's packed weights, scales and zero
points sit back to back in one buffer, so the shared q4 matmul format reads them at word offsets
from a single binding. `pieces` maps each byte range of model_q4.onnx_data to a buffer and a
destination offset, in file order, so the page can stream the file once and write each chunk
straight to the GPU. Small tensors stored inside the .onnx graph (the per-head q/k norms) go into
`inline` as numbers.

The token table and the tied lm_head are the same bytes stored twice (checked below when the data
file is present); only the embedding copy is uploaded, and the lm_head matmul reads it.
"""
import json, sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
MD = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "model"
ROPE_ROWS = 4096  # positions whose cos/sin rows are uploaded; the engine's context limit

m = onnx.load(MD / "model_q4.onnx", load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}
cfg = json.loads((MD / "config.json").read_text())


def ext(name):
    ed = {e.key: e.value for e in inits[name].external_data}
    return int(ed["offset"]), int(ed["length"])


buffers, pieces, inline = {}, [], []


def place(buf, name, length=None):
    """Append initializer `name` to buffer `buf`; return its byte offset there."""
    t = inits[name]
    dst = buffers.setdefault(buf, 0)
    if t.external_data:
        off, ln = ext(name)
        ln = ln if length is None else length
        pieces.append({"offset": off, "length": ln, "buf": buf, "dst": dst})
    else:
        arr = numpy_helper.to_array(t).astype(np.float32).ravel()
        ln = arr.size * 4
        inline.append({"buf": buf, "dst": dst, "f32": arr.tolist()})
    assert ln % 4 == 0, name
    buffers[buf] = dst + ln
    return dst


def q4(buf, stem):
    """A MatMulNBits weight: word offsets of its packed weights, scales and zero points."""
    q = place(buf, stem + "_quant")
    s = place(buf, stem + "_scales")
    z = place(buf, stem + "_zp")
    node = next(n for n in g.node if n.op_type == "MatMulNBits" and n.input[1] == stem + "_quant")
    a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
    assert a["bits"] == 4 and a["block_size"] == 32
    return {"buf": buf, "q": q // 4, "s": s // 4, "z": z // 4, "K": a["K"], "N": a["N"]}


def mm(i, part, name):
    return q4(f"L{i}.{part}.{name}", f"model_layers_{i}_{part}_{name}_MatMul_weight")


def vec(buf, name):
    return place(buf, name) // 4


# Embedding: GatherBlockQuantized, 4-bit, block 32 along the hidden axis, f32 scales, packed zero points.
emb = {"buf": "embed", "q": place("embed", "model_embed_tokens_weight_quant") // 4,
       "s": place("embed", "model_embed_tokens_weight_scales") // 4,
       "z": place("embed", "model_embed_tokens_weight_zp") // 4,
       "K": cfg["hidden_size"], "N": cfg["vocab_size"]}

# RoPE tables (positions x 32), the first ROPE_ROWS rows of each.
half = 32
rope = {"buf": "rope",
        "cos": place("rope", "cos_cache", ROPE_ROWS * half * 4) // 4,
        "sin": place("rope", "sin_cache", ROPE_ROWS * half * 4) // 4,
        "rows": ROPE_ROWS}

layers = []
for i, kind in enumerate(cfg["layer_types"]):
    L = {"kind": kind}
    nb = f"L{i}.norms"
    L["op_norm"] = vec(nb, f"model.layers.{i}.operator_layernorm.weight")
    L["ffn_norm"] = vec(nb, f"model.layers.{i}.ffn_layernorm.weight")
    L["norms"] = nb
    if kind == "conv":
        L["in_proj"] = mm(i, "conv", "in_proj")
        L["out_proj"] = mm(i, "conv", "out_proj")
        L["conv_w"] = vec(nb, f"model.layers.{i}.conv.conv.weight")   # [1024, 1, 3]
    else:
        L["q"] = mm(i, "attn", "q_proj")
        L["k"] = mm(i, "attn", "k_proj")
        L["v"] = mm(i, "attn", "v_proj")
        L["o"] = mm(i, "attn", "o_proj")
        L["q_norm"] = vec(nb, f"model.layers.{i}.attn.q_norm.layernorm.weight")
        L["k_norm"] = vec(nb, f"model.layers.{i}.attn.k_norm.layernorm.weight")
    L["gate"] = mm(i, "mlp", "gate_proj")
    L["up"] = mm(i, "mlp", "up_proj")
    L["down"] = mm(i, "mlp", "down_proj")
    layers.append(L)

final = vec("final", "model.layers.16.final_norm_layernorm.weight")

# The final norm is exported as SkipSimplifiedLayerNormalization(x, skip=x): it normalizes 2x,
# which equals normalizing x with epsilon / 4.
eps = cfg["norm_eps"]
gqa = next(n for n in g.node if n.op_type == "GroupQueryAttention")
ga = {x.name: onnx.helper.get_attribute_value(x) for x in gqa.attribute}
assert ga["rotary_interleaved"] == 0 and ga["local_window_size"] == -1

pieces.sort(key=lambda p: p["offset"])
for a, b in zip(pieces, pieces[1:]):
    assert a["offset"] + a["length"] <= b["offset"], "overlapping pieces"

data = MD / "model_q4.onnx_data"
if data.exists() and data.stat().st_size == max(o + ln for o, ln in (ext(n) for n in inits if inits[n].external_data)):
    raw = np.memmap(data, dtype=np.uint8, mode="r")
    o1, l1 = ext("model_embed_tokens_weight_quant")
    o2, l2 = ext("model_embed_tokens_weight_quant_matmul")
    assert l1 == l2 and np.array_equal(raw[o1:o1 + l1], raw[o2:o2 + l2]), "lm_head is not the token table"
    print("checked: lm_head weights equal the token table")

out = {
    "source": "onnx-community/LFM2.5-350M-ONNX onnx/model_q4.onnx_data",
    "hidden": cfg["hidden_size"], "ffn": layers[0]["gate"]["N"], "vocab": cfg["vocab_size"],
    "heads": ga["num_heads"], "kv_heads": ga["kv_num_heads"], "head_dim": cfg["hidden_size"] // ga["num_heads"],
    "scale": ga["scale"], "eps": eps, "final_eps": eps / 4, "conv_L": cfg["conv_L_cache"],
    "bos": cfg["bos_token_id"], "eos": cfg["eos_token_id"],
    "buffers": buffers, "pieces": pieces, "inline": inline,
    "embed": emb, "rope": rope, "final_norm": final, "layers": layers,
    "file_bytes": max(p["offset"] + p["length"] for p in pieces),
}
(ROOT / "web").mkdir(exist_ok=True)
(ROOT / "web" / "manifest.json").write_text(json.dumps(out, separators=(",", ":")))
needed = sum(p["length"] for p in pieces)
print(f"{len(buffers)} buffers, {len(pieces)} pieces, {needed / 1e6:.1f} MB to upload, largest buffer "
      f"{max(buffers.values()) / 1e6:.1f} MB")
