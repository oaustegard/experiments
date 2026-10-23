"""Write the engine's weight manifest for the full 24-layer EmbeddingGemma 2 text model
(onnx-community/embeddinggemma-2-ONNX, onnx/model_q4.onnx).

Usage: python3 -I run/manifest_full.py [--model out/full/model_q4.onnx] [--out web/manifest_full.json]

Same schema as run/manifest.py, so the engine and run/ref.py read either. This export differs from
the k=4 cut in ways the manifest absorbs:
- q/k/v and gate/up are fused matmuls; the manifest describes each part as a slice of rows
  (MatMulNBits keeps every output column's blocks, scales and zero points contiguous).
- q/k/v share one norm weight tiled over heads; v's part is all ones and is recorded as such.
- 4-bit token table (GatherBlockQuantized) with sqrt(512) already folded in.
- Layers 5, 11, 17, 23 are full attention: head_dim 512, one KV head, RoPE theta 1e6.
- The exporter folds each layer scalar into the next RMSNorm's epsilon (rms(s x, e) =
  rms(x, e / s^2)) and the PLE scale into the PLE norm's epsilon. The engine applies the scalar to
  the residual stream, so each norm's eps here is the one that holds for the engine's input.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent


def build(path):
    m = onnx.load(str(path), load_external_data=False)
    g = m.graph
    init = {i.name: i for i in g.initializer}

    def ext(name):
        e = {x.key: x.value for x in init[name].external_data}
        return {"offset": int(e["offset"]), "length": int(e["length"]), "dims": list(init[name].dims),
                "dtype": onnx.TensorProto.DataType.Name(init[name].data_type)}

    def inline(name):
        return numpy_helper.to_array(init[name])

    def attr(n):
        return {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

    def rows(w, n0, n):
        """Rows [n0, n0 + n) of a MatMulNBits weight."""
        nb = w["K"] // w["block"]
        zrow = (nb + 1) // 2
        sl = lambda t, per: {**t, "offset": t["offset"] + n0 * per, "length": n * per,  # noqa: E731
                             "dims": [n] + t["dims"][1:]}
        return {**w, "N": n, "q": sl(w["q"], w["K"] // 2), "scales": sl(w["scales"], nb * 4), "zp": sl(w["zp"], zrow)}

    def part(t, start, n):
        return {**t, "offset": t["offset"] + start * 4, "length": n * 4, "dims": [n]}

    t, layers, eps_graph, fused, qkv_split = {}, {}, {}, {}, {}
    for n in g.node:
        parts = n.name.split("/")
        li = int(parts[2].split(".")[1]) if parts[2].startswith("layers.") else None
        L = layers.setdefault(li, {}) if li is not None else None
        a = attr(n)
        if n.op_type == "MatMulNBits":
            w = {"q": ext(n.input[1]), "scales": ext(n.input[2]), "zp": ext(n.input[3]),
                 "N": a["N"], "K": a["K"], "block": a["block_size"], "bits": a["bits"]}
            role = parts[-2]
            if li is None:
                t[{"per_layer_model_projection": "ple"}.get(role, role)] = w
            else:
                fused[(li, role)] = w
                if role not in ("qkv_proj", "gate_up_proj"):
                    L[role] = w
        elif n.op_type in ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"):
            wname = n.input[1] if n.op_type == "SimplifiedLayerNormalization" else n.input[2]
            role = parts[-2]
            if role == "qkv_norm":
                continue
            spec = {**ext(wname), "graph_eps": a["epsilon"]}
            if li is None:
                t[role] = spec
            else:
                L[role] = spec
        elif n.op_type == "Mul" and parts[-2] == "qkv_norm":
            L["qkv_norm"] = ext(n.input[1])
            L["_qkv_norm"] = n.input[1]
        elif n.op_type == "Mul" and parts[-1] == "Mul_layer_scalar":
            L["layer_scalar"] = float(inline(n.input[1]).reshape(-1)[0])
        elif n.op_type == "Split" and parts[-2] == "qkv_proj":
            qkv_split[li] = [int(x) for x in inline(n.input[1])]
        elif n.op_type == "RotaryEmbedding" and parts[-2] == "q_rotary":
            L["rope"] = n.input[2].split("/")[3]  # full_attention | sliding_attention
        elif n.op_type == "Reshape" and parts[-2] == "q" and parts[-1] == "Reshape":
            shape = [int(x) for x in inline(n.input[1])]  # [0, 0, kv heads, group, head_dim]
            L["n_kv_heads"], L["head_dim"] = shape[2], shape[4]
            L["n_heads"] = shape[2] * shape[3]
        elif n.op_type == "Reshape" and parts[-2] == "ple" and parts[-1] == "Reshape":
            shape = [int(x) for x in inline(n.input[1])]
            t["ple_layers"], t["ple_dim"] = shape[2], shape[3]
    for k in list(fused):
        li, role = k
        L, w = layers[li], fused[k]
        if role == "qkv_proj":
            qn, kn, vn = qkv_split[li]
            L["q_proj"], L["k_proj"], L["v_proj"] = rows(w, 0, qn), rows(w, qn, kn), rows(w, qn + kn, vn)
            hd = L["head_dim"]
            nw = L.pop("qkv_norm")
            L.pop("_qkv_norm")
            L["q_norm"], L["k_norm"] = part(nw, 0, hd), part(nw, qn, hd)  # tiled over heads (checked below)
        elif role == "gate_up_proj":
            half = w["N"] // 2
            L["gate_proj"], L["up_proj"] = rows(w, 0, half), rows(w, half, half)

    # Each norm's eps for the engine's input: the graph eps times the square of the layer scalar
    # the engine has already applied (the previous layer's, for input_layernorm).
    out_layers = [layers[i] for i in sorted(layers)]
    for i, L in enumerate(out_layers):
        L.setdefault("layer_scalar", 1.0)  # the last layer's is folded into the final norm's eps
        L["attn_scale"] = 1.0
        for k, v in L.items():
            if isinstance(v, dict) and "graph_eps" in v:
                s = out_layers[i - 1]["layer_scalar"] if (k == "input_layernorm" and i) else 1.0
                v["eps"] = float(np.float32(v.pop("graph_eps")) * np.float32(s) ** 2)
    for k in ("norm", "per_layer_projection_norm"):
        t[k]["eps"] = float(t[k].pop("graph_eps"))
    t["per_layer_projection_norm"]["eps"] = float(np.float32(t["per_layer_projection_norm"]["eps"]) / np.float32(512))
    t["ple_scale"] = float(np.float32(512) ** np.float32(-0.5))
    rope = {}
    for kind in ("sliding_attention", "full_attention"):
        rope[kind] = inline(f"model.rotary_emb.{kind}_inv_freq").astype(np.float32).reshape(-1).tolist() \
            if not init[f"model.rotary_emb.{kind}_inv_freq"].external_data else None
    tbl = {"kind": "q4", "codes": ext("model_embed_tokens_weight_quant"), "scales": ext("model_embed_tokens_weight_scales"),
           "zp": ext("model_embed_tokens_weight_zp"), "bits": 4, "block": 32}
    out = {"source": path.name, "data": path.name + "_data", "hidden": 512, "table": tbl, "embed_scale": 1.0,
           "eps": 1e-6, "rope": rope, "layers": out_layers, **t}
    return out


def fill_external(man, data):
    """Inline any RoPE frequency table stored in the external file, and check the tiled q/k norms."""
    for kind, v in man["rope"].items():
        if v is None:
            raise SystemExit(f"{kind} inv_freq not inline; pass it through the manifest")
    for L in man["layers"]:
        for k in ("q_norm", "k_norm"):
            w = L[k]
            n = (L["n_heads"] if k == "q_norm" else L["n_kv_heads"]) * L["head_dim"]
            full = np.frombuffer(data, np.float32, n, w["offset"]).reshape(-1, L["head_dim"])
            assert np.allclose(full, full[0]), f"{k} not tiled over heads"
        qn, kn = L["q_proj"]["N"], L["k_proj"]["N"]
        v = np.frombuffer(data, np.float32, L["v_proj"]["N"], L["q_norm"]["offset"] + (qn + kn) * 4)
        assert np.allclose(v, 1), "v norm weight is not ones"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "out" / "full" / "model_q4.onnx"))
    ap.add_argument("--out", default=str(ROOT / "web" / "manifest_full.json"))
    a = ap.parse_args()
    path = Path(a.model)
    man = build(path)
    # full_attention inv_freq lives in the external file; read it from there
    m = onnx.load(str(path), load_external_data=False)
    init = {i.name: i for i in m.graph.initializer}
    data = (path.parent / man["data"]).read_bytes()
    for kind in man["rope"]:
        if man["rope"][kind] is None:
            e = {x.key: x.value for x in init[f"model.rotary_emb.{kind}_inv_freq"].external_data}
            man["rope"][kind] = np.frombuffer(data, np.float32, int(e["length"]) // 4, int(e["offset"])).tolist()
    fill_external(man, data)
    Path(a.out).write_text(json.dumps(man, indent=1))
    L5 = man["layers"][5]
    print("layers", len(man["layers"]), "L5", L5["head_dim"], L5["n_heads"], L5["n_kv_heads"], L5["rope"],
          "eps", [round(L["input_layernorm"]["eps"] * 1e6, 4) for L in man["layers"]][:6], man["norm"]["eps"],
          man["per_layer_projection_norm"]["eps"], "rope", {k: len(v) for k, v in man["rope"].items()})


if __name__ == "__main__":
    main()
