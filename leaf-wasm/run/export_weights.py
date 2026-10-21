"""Pack mdbr-leaf-ir's safetensors into the single weights file web/leaf.js reads.

Usage: python3 -I run/export_weights.py        (reads out/hf/, writes out/leaf-ir-f32.bin, out/leaf-ir-i8.bin)
       python3 -I run/export_weights.py --keep KEPT_OLD_IDS.json --tag k16000   (+ -t8.bin and -t4.bin)

Layout changes from the checkpoint, all exact in fp32:
  - Linear weights transposed to [in][out], the kernel's W[K][N] layout.
  - query/key/value fused into one [384][1152] matrix (+ bias), one matmul per layer.
  - token_type_embeddings[0] folded into position_embeddings: every input is type 0.
File: b"LEAF", u32 version, u32 header length, header JSON, then tensors at 16-byte
offsets. i8 tensors carry f32 scales, symmetric: per output column for matrices [K][N]
(axis 1), per row for the token table (axis 0). Biases and LayerNorm stay f32.
i4 (token table only, the t4 of ettin-native/run/export.py): symmetric int4 in groups of 32
along the row, packed two per byte [V][H/2] (low nibble = even column, stored value + 8),
f32 scales [V][H/32].
--keep prunes the vocabulary: table rows = table_full[kept_old_ids]; the matching vocab.txt
(line n = new id) ships beside the weights. Built by ettin-native for leaf-ir k16000.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "out" / "hf"


def tensors():
    w = load_file(str(HF / "model.safetensors"))
    d = load_file(str(HF / "2_Dense" / "model.safetensors"))
    cfg = json.loads((HF / "config.json").read_text())
    t = {"tok_emb": w["embeddings.word_embeddings.weight"],
         "pos_emb": w["embeddings.position_embeddings.weight"] + w["embeddings.token_type_embeddings.weight"][0],
         "emb_ln_g": w["embeddings.LayerNorm.weight"], "emb_ln_b": w["embeddings.LayerNorm.bias"]}
    for i in range(cfg["num_hidden_layers"]):
        p = f"encoder.layer.{i}."
        a = p + "attention."
        t[f"l{i}.qkv_w"] = np.concatenate([w[a + f"self.{x}.weight"] for x in ("query", "key", "value")]).T
        t[f"l{i}.qkv_b"] = np.concatenate([w[a + f"self.{x}.bias"] for x in ("query", "key", "value")])
        t[f"l{i}.o_w"] = w[a + "output.dense.weight"].T
        t[f"l{i}.o_b"] = w[a + "output.dense.bias"]
        t[f"l{i}.ln1_g"] = w[a + "output.LayerNorm.weight"]
        t[f"l{i}.ln1_b"] = w[a + "output.LayerNorm.bias"]
        t[f"l{i}.up_w"] = w[p + "intermediate.dense.weight"].T
        t[f"l{i}.up_b"] = w[p + "intermediate.dense.bias"]
        t[f"l{i}.down_w"] = w[p + "output.dense.weight"].T
        t[f"l{i}.down_b"] = w[p + "output.dense.bias"]
        t[f"l{i}.ln2_g"] = w[p + "output.LayerNorm.weight"]
        t[f"l{i}.ln2_b"] = w[p + "output.LayerNorm.bias"]
    t["dense_w"] = d["linear.weight"].T
    t["dense_b"] = d["linear.bias"]
    config = {"hidden": cfg["hidden_size"], "heads": cfg["num_attention_heads"], "layers": cfg["num_hidden_layers"],
              "ffn": cfg["intermediate_size"], "vocab": cfg["vocab_size"], "max_pos": cfg["max_position_embeddings"],
              "out_dim": int(d["linear.weight"].shape[0]), "eps": cfg["layer_norm_eps"]}
    return {k: np.ascontiguousarray(v, np.float32) for k, v in t.items()}, config


def quantize(a, axis, levels=127):
    """Symmetric, codes in [-levels, levels] stored as int8. Fewer levels = lower entropy, so the
    file brotli/gzip-compresses smaller on the wire; the loader is unchanged (q * scale)."""
    s = np.abs(a).max(axis=1 - axis if a.ndim == 2 else 0) / levels
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    q = np.round(a / (s[None, :] if axis == 1 else s[:, None])).clip(-levels, levels).astype(np.int8)
    return q, s


def quantize4(a, group=32):
    V, H = a.shape
    g = a.reshape(V, H // group, group)
    s = np.abs(g).max(axis=2) / 7.0
    s = np.where(s == 0, 1.0, s).astype(np.float32)
    q = (np.round(g / s[..., None]).clip(-7, 7).astype(np.int8) + 8).reshape(V, H).astype(np.uint8)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), s


def write(path, t, config, int8, table_bits=8, levels=127):
    blobs, header = [], {"config": config, "tensors": {}}
    off = 0

    def add(arr):
        nonlocal off
        pos = off
        b = arr.tobytes()
        blobs.append((pos, b))
        off = (pos + len(b) + 15) // 16 * 16
        return pos

    for name, a in t.items():
        axis = 0 if name == "tok_emb" else 1
        if int8 and name == "tok_emb" and table_bits == 4:
            q, s = quantize4(a)
            header["tensors"][name] = {"dtype": "i4", "shape": list(a.shape), "offset": add(q),
                                       "scale": add(s), "group": 32}
        elif int8 and a.ndim == 2 and name != "pos_emb":
            q, s = quantize(a, axis, levels)
            header["tensors"][name] = {"dtype": "i8", "shape": list(a.shape), "offset": add(q),
                                       "scale": add(s), "axis": axis}
        else:
            header["tensors"][name] = {"dtype": "f32", "shape": list(a.shape), "offset": add(a)}
    h = json.dumps(header, separators=(",", ":")).encode()
    start = (12 + len(h) + 15) // 16 * 16
    buf = bytearray(start + off)
    buf[:12] = b"LEAF" + struct.pack("<II", 1, len(h))
    buf[12:12 + len(h)] = h
    for pos, b in blobs:
        buf[start + pos:start + pos + len(b)] = b
    path.write_bytes(bytes(buf))
    print(path.name, f"{len(buf) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="kept_old_ids.json: new token id -> original id")
    ap.add_argument("--tag", default="k")
    ap.add_argument("--levels", type=int, nargs="*", help="also write out/leaf-ir-i8-L<n>.bin per level count")
    a = ap.parse_args()
    t, config = tensors()
    if a.levels:
        for n in a.levels:
            write(ROOT / "out" / f"leaf-ir-i8-L{n}.bin", t, config, int8=True, levels=n)
        return
    if not a.keep:
        write(ROOT / "out" / "leaf-ir-f32.bin", t, config, int8=False)
        write(ROOT / "out" / "leaf-ir-i8.bin", t, config, int8=True)
        return
    keep = np.array(json.loads(Path(a.keep).read_text()), np.int64)
    t["tok_emb"] = np.ascontiguousarray(t["tok_emb"][keep])
    config = {**config, "vocab": len(keep)}
    for bits in (8, 4):
        write(ROOT / "out" / f"leaf-ir-{a.tag}-t{bits}.bin", t, config, int8=True, table_bits=bits)


if __name__ == "__main__":
    main()
