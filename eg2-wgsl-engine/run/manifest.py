"""Write the engine's weight manifest for a shipped k-layer EG2 cut (model_t3w.onnx).

Usage: python3 -I run/manifest.py [--model out/model/model_t3w.onnx] [--out web/manifest.json]

The engine reads the same `model_t3w.onnx_data` file onnxruntime-web loads, so the page downloads
the weights once for either path. This script walks the ONNX graph in node order, names every
initializer by its role (q/k/v/o, gate/up/down, PLE, norms) and records its byte offset and
length in the external-data file, plus the constants the graph keeps inline (RoPE inverse
frequencies, layer scalars, embedding scale, eps).
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
    const = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in g.node
             if n.op_type == "Constant" and n.attribute[0].type == onnx.AttributeProto.TENSOR}

    def ext(name):
        e = {x.key: x.value for x in init[name].external_data}
        return {"offset": int(e["offset"]), "length": int(e["length"]), "dims": list(init[name].dims),
                "dtype": onnx.TensorProto.DataType.Name(init[name].data_type)}

    def value(name):
        return numpy_helper.to_array(init[name]) if name in init else const[name]

    t = {}
    layers = {}
    for n in g.node:
        scope = n.output[0]
        li = None
        if scope.startswith("/tm/layers."):
            li = int(scope.split("/")[2].split(".")[1])
            L = layers.setdefault(li, {})
        if n.op_type == "MatMulNBits":
            a = {x.name: x.i for x in n.attribute}
            w = {"q": ext(n.input[1]), "scales": ext(n.input[2]), "zp": ext(n.input[3]),
                 "N": a["N"], "K": a["K"], "block": a["block_size"], "bits": a["bits"]}
            role = scope.split("/")[-2] if li is not None else scope.split("/")[2]
            (L if li is not None else t)[role] = w
        elif n.op_type == "Mul" and n.input[1] in init and init[n.input[1]].external_data:
            role = scope.split("/")[-2]  # .../<norm>/Mul_1_output_0
            if li is not None:
                L[role] = ext(n.input[1])
            else:
                t[role] = ext(n.input[1])
        elif n.op_type == "Mul" and n.input[1].endswith("layer_scalar"):
            L["layer_scalar"] = float(value(n.input[1]).reshape(-1)[0])
        elif n.op_type == "Mul" and scope == "/tm/embed_tokens/Mul_output_0":
            t["embed_scale"] = float(value(n.input[1]))
        elif n.op_type == "Mul" and scope == "/tm/ple/Mul_output_0":
            t["ple_scale"] = float(value(n.input[1]))
        elif n.op_type == "Mul" and scope == "/tm/rotary_emb/Mul_output_0":
            t["inv_freq"] = value(n.input[1]).astype(np.float32).reshape(-1).tolist()
        elif n.op_type == "Mul" and scope.endswith("self_attn/Mul_8_output_0"):
            L["attn_scale"] = float(value(n.input[1]))
        elif n.op_type == "Add" and scope.endswith("norm/Add_output_0") and "eps" not in t:
            t["eps"] = float(value(n.input[1]))
        elif n.op_type == "Concat" and scope == "/tm/ple/Concat_output_0":
            t["ple_layers"], t["ple_dim"] = int(value(n.input[2])[0]), int(value(n.input[3])[0])
        elif n.op_type == "Concat" and li is not None and scope.endswith("self_attn/Concat_output_0"):
            L["head_dim"] = int(value(n.input[3])[0])
    tbl = {k: ext(f"/tm/embed_tokens/t3_{k}") for k in ("codes", "scales", "zp")}
    tbl["bits"] = 3
    tbl["block"] = 32
    out = {"source": path.name, "data": path.name + "_data", "hidden": 512, "table": tbl,
           "layers": [layers[i] for i in sorted(layers)], **t}
    for L in out["layers"]:
        L["n_heads"] = L["q_proj"]["N"] // L["head_dim"]
        L["n_kv_heads"] = L["k_proj"]["N"] // L["head_dim"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "out" / "model" / "model_t3w.onnx"))
    ap.add_argument("--out", default=str(ROOT / "web" / "manifest.json"))
    a = ap.parse_args()
    man = build(Path(a.model))
    Path(a.out).write_text(json.dumps(man, indent=1))
    print("layers", len(man["layers"]), "keys", sorted(man["layers"][0]), sorted(k for k in man if k != "layers"))


if __name__ == "__main__":
    main()
