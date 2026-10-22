"""Put a LEAF weights file back into MongoDB's fp32 ONNX graph, to score it with ORT.

Usage: python3 -I run/patch_onnx.py out/leaf-ir-i8.bin out/onnx-i8deq

The JS engine dequantizes int8 weights to f32 and computes in f32, and matches ORT fp32 to
cosine 0.99999993 on the fixtures (results/parity.json). So ORT running the fp32 graph with
the dequantized weights computes what the engine computes, and can embed the 61,686-post
corpus in minutes instead of hours of single-threaded JS. Self-check: patching
out/leaf-ir-f32.bin must leave every initializer bit-identical.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "out" / "hf"


def load_leaf(path):
    buf = Path(path).read_bytes()
    assert buf[:4] == b"LEAF"
    (hlen,) = struct.unpack("<I", buf[8:12])
    header = json.loads(buf[12:12 + hlen])
    base = (12 + hlen + 15) // 16 * 16
    out = {}
    for name, t in header["tensors"].items():
        n = int(np.prod(t["shape"]))
        if t["dtype"] == "f32":
            a = np.frombuffer(buf, np.float32, n, base + t["offset"]).reshape(t["shape"])
        else:
            q = np.frombuffer(buf, np.int8, n, base + t["offset"]).reshape(t["shape"]).astype(np.float32)
            s = np.frombuffer(buf, np.float32, t["shape"][t["axis"]], base + t["scale"])
            a = q * (s[None, :] if t["axis"] == 1 else s[:, None])  # f32 product, same as the JS loader
        out[name] = a.astype(np.float32)
    return out, header["config"]


def onnx_tensors(w, cfg):
    """ONNX initializer name -> array, inverting export_weights.py's layout changes."""
    H = cfg["hidden"]
    t = {".model.embeddings.word_embeddings.weight": w["tok_emb"],
         "/model/st/dense_1.weight": w["dense_w"], "/model/st/dense_1.bias": w["dense_b"]}
    for i in range(cfg["layers"]):
        p = f"model.layers.{i}."
        for j, x in enumerate("qkv"):
            t[p + f"attn.{x}_proj.MatMul.weight"] = w[f"l{i}.qkv_w"][:, j * H:(j + 1) * H]
            t[p + f"attn.{x}_proj.Add.bias"] = w[f"l{i}.qkv_b"][j * H:(j + 1) * H]
        t[p + "attn.o_proj.MatMul.weight"] = w[f"l{i}.o_w"]
        t[p + "attn.o_proj.Add.bias"] = w[f"l{i}.o_b"]
        t[p + "mlp.fc1.MatMul.weight"] = w[f"l{i}.up_w"]
        t[p + "mlp.fc1.Add.bias"] = w[f"l{i}.up_b"]
        t[p + "mlp.fc2.MatMul.weight"] = w[f"l{i}.down_w"]
        t[p + "mlp.fc2.Add.bias"] = w[f"l{i}.down_b"]
    return t


def main():
    src, dst = sys.argv[1], Path(sys.argv[2])
    w, cfg = load_leaf(src)
    new = onnx_tensors(w, cfg)
    m = onnx.load(str(HF / "onnx" / "model.onnx"))
    changed, maxdiff = 0, 0.0
    for init in m.graph.initializer:
        if init.name in new:
            old = numpy_helper.to_array(init)
            a = np.ascontiguousarray(new[init.name])
            assert a.shape == old.shape, (init.name, a.shape, old.shape)
            maxdiff = max(maxdiff, float(np.abs(a - old).max()))
            init.CopyFrom(numpy_helper.from_array(a, init.name))
            changed += 1
    print(f"replaced {changed} initializers, max |new - original| = {maxdiff:.3g}")
    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst / "onnx" / "model.onnx"), save_as_external_data=True, location="model.onnx_data")
    shutil.copy(HF / "tokenizer.json", dst / "tokenizer.json")


if __name__ == "__main__":
    main()
