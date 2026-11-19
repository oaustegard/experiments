"""Pack a .traineddata LSTM into the web engine's format (manifest.json + weights.bin).

    python3 run/pack.py models/best/eng.traineddata web/models/best-eng [--weights f32|f16|q8|q6|q4]

Tensors: w_conv [16, 10], w_y / w_x1 / w_x2 / w_x3 [4*ns, ni+ns+1] (gate blocks CI, GI, GF1, GO),
w_out [C, ns3+1]. An int8 model (tessdata_fast) keeps its own codes and per-row scales and runs
in Tesseract's int mode. A float model (tessdata_best) can be stored as f32 or f16, or packed to
q8/q6/q4 per-row codes with scale max|row|/qmax; the engine expands every format to f32 at load,
so packing changes the download and the weights, not the kernels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tessgpu.traineddata import load  # noqa: E402

QMAX = {"q8": 127, "q6": 31, "q4": 7}


def tensors(m):
    kids = m.network.children
    gates = lambda n: [n.gates[k] for k in ("CI", "GI", "GF1", "GO")]  # noqa: E731
    out = {"w_conv": [kids[1].children[1].weights]}
    for name, node in (("w_y", kids[3].children[0]), ("w_x1", kids[4]),
                       ("w_x2", kids[5].children[0]), ("w_x3", kids[6])):
        out[name] = gates(node)
    out["w_out"] = [kids[7].weights]
    return out


def quantize_rows(w, qmax):
    scale = np.abs(w).max(axis=1) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -qmax, qmax).astype(np.int8)
    return q, scale


def fake_quant(w, fmt):
    """The float weights a packed format decodes to (used by the accuracy sweep too)."""
    if fmt == "f32":
        return w.astype(np.float32)
    if fmt == "f16":
        return w.astype(np.float16).astype(np.float32)
    q, s = quantize_rows(w, QMAX[fmt])
    return (q.astype(np.float32) * s[:, None]).astype(np.float32)


def pack(model_path, out_dir, fmt="f32"):
    m = load(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    entries = {}

    def put(arr):
        nonlocal blob
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob += arr.tobytes()
        return off

    for name, mats in tensors(m).items():
        if m.int_mode:
            q = np.concatenate([x.q for x in mats]).astype(np.int8)
            s = np.concatenate([x.scale for x in mats]).astype(np.float32)
            entries[name] = {"dtype": "i8", "shape": list(q.shape), "offset": put(q),
                             "scale_offset": put(s), "int_mode": True}
            continue
        w = np.concatenate([x.w for x in mats])
        if fmt == "f32":
            entries[name] = {"dtype": "f32", "shape": list(w.shape), "offset": put(w.astype(np.float32))}
        elif fmt == "f16":
            entries[name] = {"dtype": "f16", "shape": list(w.shape), "offset": put(w.astype(np.float16))}
        else:
            q, s = quantize_rows(w, QMAX[fmt])
            # codes stored one per byte; q6/q4 ranges only (bit-packing is a download concern,
            # gzip on the wire recovers most of it and keeps the loader simple)
            entries[name] = {"dtype": "i8", "shape": list(q.shape), "offset": put(q),
                             "scale_offset": put(s.astype(np.float32)), "int_mode": False}
    kids = m.network.children
    ns = {"y": kids[3].children[0].ns, "x1": kids[4].ns, "x2": kids[5].children[0].ns, "x3": kids[6].ns}
    ni = {"y": 16, "x1": ns["y"], "x2": ns["x1"], "x3": ns["x2"]}
    man = {
        "format": "tesseract-gpu/1", "source": Path(model_path).name,
        "variant": Path(model_path).parent.name, "weights": "int8-native" if m.int_mode else fmt,
        "version": m.version, "spec": m.network_str, "H": m.input_height, "C": m.num_classes,
        "int_mode": m.int_mode, "simple_text": m.simple_text, "null_char": m.null_char, "sample_iteration": m.sample_iteration,
        "ns": ns, "ni": ni, "unichars": m.unicharset, "codes": [list(c) for c in m.recoder],
        "tensors": entries, "bytes": len(blob),
    }
    (out_dir / "weights.bin").write_bytes(bytes(blob))
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--weights", default="f32", choices=["f32", "f16", "q8", "q6", "q4"])
    a = ap.parse_args()
    man = pack(a.model, a.out, a.weights)
    print(a.out, man["weights"], man["bytes"], "bytes", man["spec"])


if __name__ == "__main__":
    main()
