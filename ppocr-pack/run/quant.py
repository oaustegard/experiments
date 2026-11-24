#!/usr/bin/env python3
"""The packed weight format for PP-OCRv6, simulated on ONNX initializers.

    python3 -I run/quant.py models/tiny_rec r63            # sizes of one arm
    python3 -I run/quant.py models/tiny_rec r63 --head t4  # CTC head at 4 bits
    python3 -I run/quant.py models/tiny_rec r63 --head t4 --out /tmp/pack   # writes weights.bin + manifest.json

The format, which the browser engines load:

  1. BatchNorm is folded into the preceding Conv (through a Squeeze of a non-batch, non-channel axis if one
     sits between them). A BatchNorm with no foldable producer stays as it is, in fp32.
  2. Every Conv weight [O, I/g, kh, kw] gets one scale per output channel (axis 0). A ConvTranspose weight
     [I, O/g, kh, kw] gets one per output channel too, which is axis 1. A MatMul weight [K, N] gets one per
     column N (axis 1). A tensor needs at least 1,024 elements to be packed.
  3. rL: s = max|w| / L over the channel, stored as fp16; codes = round(w / s16) clipped to +-L, stored as int8
     whatever L is (r127, r63, r31). Dequantised weight = code * s16.
  4. t4 (the CTC head only): [K, N] weight in groups of 32 along K (the last group of a column may be
     shorter), one fp16 scale per group and column, s = max|w| / 7, codes round(w / s16) in +-7.
  5. Biases, norm parameters, any float tensor under 1,024 elements, and the biases folding creates stay fp32.
     Integer initializers (shapes, axes) are graph structure and are not counted.
  f16 (a comparison arm, not part of the format): the packable tensors are stored as fp16 instead.

Byte stream (the size accounting): the packable tensors in initializer order, each as its codes then its fp16
scales (int8 codes row-major in ONNX layout; t4 codes row-major [K, N] offset by +8 into nibbles, two
consecutive elements per byte, low nibble first; t4 scales fp16 [ceil(K/32), N] row-major), then every fp32
tensor in initializer order as raw little-endian float32.

Fake-quantisation: `pack` returns the ONNX model with every packable initializer replaced by its dequantised
fp32 value, so onnxruntime runs the arithmetic the browser engines would, up to float summation order.
"""
import argparse
import copy
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MIN_ELEMS = 1024
GROUP = 32
HEAD_BITS = {"t4": 7}


# ------------------------------------------------------------------ graph helpers


def _maps(g):
    prod, cons = {}, {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return prod, cons


def _squeeze_axes(n):
    for a in n.attribute:
        if a.name == "axes":
            return list(a.ints)
    return None


def fold_batchnorm(model):
    """Fold every foldable BatchNormalization into its Conv, in place. Returns a report dict."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    prod, cons = _maps(g)
    uses = {}
    for n in g.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    folded, kept, drop_nodes, drop_inits = [], [], [], set()
    for bn in [n for n in g.node if n.op_type == "BatchNormalization"]:
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        chain, x = [], bn.input[0]
        p = prod.get(x)
        if p is not None and p.op_type == "Squeeze":
            ax = _squeeze_axes(p)
            if ax is None or any(a < 2 for a in ax) or len(cons[x]) != 1:  # negative axes too: unknown rank
                kept.append((bn.name, "squeeze on batch/channel/unknown axis"))
                continue
            chain.append(p)
            x = p.input[0]
            p = prod.get(x)
        if p is None or p.op_type != "Conv" or len(cons[x]) != 1:
            kept.append((bn.name, f"producer {p.op_type if p is not None else 'graph input'}"))
            continue
        if len(bn.output) != 1 or any(t not in inits for t in bn.input[1:]):
            kept.append((bn.name, "non-constant parameters or extra outputs"))
            continue
        wname = p.input[1]
        if wname not in inits or uses[wname] != 1:
            kept.append((bn.name, "shared or non-constant conv weight"))
            continue
        gamma, beta, mean, var = (numpy_helper.to_array(inits[t]).astype(np.float64) for t in bn.input[1:])
        k = gamma / np.sqrt(var + eps)
        w = numpy_helper.to_array(inits[wname]).astype(np.float64)
        bias = numpy_helper.to_array(inits[p.input[2]]).astype(np.float64) if len(p.input) > 2 else np.zeros_like(mean)
        w2 = (w * k.reshape(-1, *([1] * (w.ndim - 1)))).astype(np.float32)
        b2 = ((bias - mean) * k + beta).astype(np.float32)
        inits[wname].CopyFrom(numpy_helper.from_array(w2, wname))
        if len(p.input) > 2:
            inits[p.input[2]].CopyFrom(numpy_helper.from_array(b2, p.input[2]))
        else:
            bname = wname + ".bn_bias"
            g.initializer.append(numpy_helper.from_array(b2, bname))
            p.input.append(bname)
        (chain[0] if chain else p).output[0] = bn.output[0]
        drop_nodes.append(bn)
        drop_inits.update(bn.input[1:])
        folded.append(bn.name)
    for n in drop_nodes:
        g.node.remove(n)
    still = {i for n in g.node for i in n.input}
    for name in drop_inits - still:
        g.initializer.remove(next(i for i in g.initializer if i.name == name))
    return {"folded": folded, "kept": kept}


def targets(model):
    """{initializer name: (op_type, quantisation axis)} for every packable weight (>= 1,024 elements)."""
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    out = {}
    for n in g.node:
        if n.op_type in ("Conv", "ConvTranspose", "MatMul") and len(n.input) > 1 and n.input[1] in inits:
            t = inits[n.input[1]]
            if t.data_type != onnx.TensorProto.FLOAT or int(np.prod(t.dims)) < MIN_ELEMS:
                continue
            if n.op_type == "MatMul" and len(t.dims) != 2:
                continue
            out[t.name] = (n.op_type, 1 if n.op_type in ("ConvTranspose", "MatMul") else 0)
    return out


# ------------------------------------------------------------------ quantisers


def quant_channel(w, axis, L):
    """Per-channel symmetric quantiser. Returns (codes int8 same shape, scales fp16 [C], dequantised fp32)."""
    wm = np.moveaxis(w.astype(np.float32), axis, 0)
    flat = wm.reshape(wm.shape[0], -1)
    s16 = (np.abs(flat).max(axis=1) / np.float32(L)).astype(np.float16)
    s = s16.astype(np.float32)
    safe = np.where(s > 0, s, np.float32(1))[:, None]
    codes = np.clip(np.rint(flat / safe), -L, L).astype(np.int8)
    deq = (codes.astype(np.float32) * s[:, None]).reshape(wm.shape)
    return np.moveaxis(codes.reshape(wm.shape), 0, axis), s16, np.moveaxis(deq, 0, axis).astype(np.float32)


def quant_t4(w):
    """4-bit codes +-7 in groups of 32 along K for a [K, N] weight. Returns (codes int8 [K,N], scales fp16 [G,N], deq)."""
    K, N = w.shape
    G = -(-K // GROUP)
    pad = np.zeros((G * GROUP, N), np.float32)
    pad[:K] = w
    grp = pad.reshape(G, GROUP, N)
    s16 = (np.abs(grp).max(axis=1) / np.float32(7)).astype(np.float16)
    s = s16.astype(np.float32)
    safe = np.where(s > 0, s, np.float32(1))[:, None, :]
    codes = np.clip(np.rint(grp / safe), -7, 7).astype(np.int8)
    deq = (codes.astype(np.float32) * s[:, None, :]).reshape(G * GROUP, N)[:K]
    return codes.reshape(G * GROUP, N)[:K], s16, np.ascontiguousarray(deq)


def nibbles(codes):
    """Offset +8, two consecutive elements per byte, low nibble first; an odd count pads with the zero code."""
    v = (codes.reshape(-1).astype(np.int16) + 8).astype(np.uint8)
    if v.size % 2:
        v = np.append(v, np.uint8(8))
    return (v[0::2] | (v[1::2] << 4)).astype(np.uint8)


# ------------------------------------------------------------------ packing


class Packed:
    def __init__(self, model, stream, manifest, sizes, werr):
        self.model, self.stream, self.manifest, self.sizes, self.werr = model, stream, manifest, sizes, werr

    def onnx_bytes(self):
        return self.model.SerializeToString()


def head_name(model):
    """The CTC head: the MatMul weight with the most elements."""
    tg = targets(model)
    mm = {k: v for k, v in tg.items() if v[0] == "MatMul"}
    inits = {i.name: i for i in model.graph.initializer}
    return max(mm, key=lambda k: int(np.prod(inits[k].dims))) if mm else None


def pack(folded, scheme, head=None):
    """folded: a ModelProto after fold_batchnorm. scheme in f32, f16, r127, r63, r31. head: None or 't4'.
    Returns Packed; `folded` is not modified."""
    assert scheme in ("f32", "f16", "r127", "r63", "r31"), scheme
    model = copy.deepcopy(folded)
    tg = targets(model)
    hname = head_name(model) if head else None
    assert head in (None, "t4") and (head is None or hname)
    parts_q, parts_rest, manifest = [], [], []
    codes_b = scale_b = rest_b = 0
    err2 = ref2 = 0.0
    for t in model.graph.initializer:
        if t.data_type != onnx.TensorProto.FLOAT:
            continue
        w = numpy_helper.to_array(t)
        if t.name in tg and scheme != "f32":
            op, axis = tg[t.name]
            if t.name == hname:
                codes, sc, deq = quant_t4(w)
                body = nibbles(codes).tobytes()
                kind = "t4"
            elif scheme == "f16":
                deq = w.astype(np.float16).astype(np.float32)
                codes, sc, body, kind = None, None, w.astype(np.float16).tobytes(), "f16"
            else:
                L = int(scheme[1:])
                codes, sc, deq = quant_channel(w, axis, L)
                body = codes.tobytes()
                kind = scheme
            sb = sc.astype("<f2").tobytes() if sc is not None else b""
            parts_q.append(body + sb)
            codes_b += len(body)
            scale_b += len(sb)
            manifest.append({"name": t.name, "op": op, "shape": list(w.shape), "scheme": kind,
                             "axis": axis if kind not in ("t4", "f16") else None,
                             "codes_bytes": len(body), "scale_bytes": len(sb)})
            err2 += float(((deq - w).astype(np.float64) ** 2).sum())
            ref2 += float((w.astype(np.float64) ** 2).sum())
            t.CopyFrom(numpy_helper.from_array(deq, t.name))
        else:
            parts_rest.append(w.astype("<f4").tobytes())
            rest_b += w.size * 4
            manifest.append({"name": t.name, "shape": list(w.shape), "scheme": "f32", "bytes": w.size * 4})
    stream = b"".join(parts_q + parts_rest)
    sizes = {"packed_bytes": len(stream), "gzip9_bytes": len(gzip.compress(stream, 9)),
             "codes_bytes": codes_b, "scale_bytes": scale_b, "fp32_rest_bytes": rest_b,
             "n_packed_tensors": len(parts_q), "n_fp32_tensors": len(parts_rest)}
    werr = {"weight_rel_rmse": (err2 / ref2) ** 0.5 if ref2 else 0.0}
    return Packed(model, stream, manifest, sizes, werr)


def load_folded(path):
    m = onnx.load(str(path))
    return m, fold_batchnorm(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("scheme")
    ap.add_argument("--head", choices=["t4"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    m, rep = load_folded(a.model_dir / "inference.onnx")
    p = pack(m, a.scheme, a.head)
    print(json.dumps({"fold": {"folded": len(rep["folded"]), "kept": rep["kept"]}, **p.sizes, **p.werr}, indent=1))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "weights.bin").write_bytes(p.stream)
        (a.out / "manifest.json").write_text(json.dumps({"scheme": a.scheme, "head": a.head, "tensors": p.manifest}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
