#!/usr/bin/env python3
"""Export the PP-OCRv6 text-line detector (ONNX) to a straight op program plus packed weights.

    python3 -I run/export_det.py                       # tiny and small, scheme r63 -> web/det/models/<name>/
    python3 -I run/export_det.py --scheme r127 tiny    # one model, another scheme
    python3 -I run/export_det.py --sizes               # byte counts for every scheme -> results/det_sizes.json

ONNX sources are read from <src>/<name>_det/inference.{onnx,yml}; <src> is --src, else $PPOCR_MODELS,
else ppocr-pack/models.

The graph is walked once in topological order. Single-consumer chains fold into one op:

  Conv [+ BatchNormalization] [+ bias] + Relu | HardSigmoid | Sigmoid | exact GELU  -> "conv"
  ReduceMean(axes=[2,3]) or GlobalAveragePool -> Conv1x1 -> Relu -> Conv1x1 -> HardSigmoid
        -> Mul(x, gate) [-> Add(x, .)]                                              -> "se"
  ConvTranspose [+ bias] [+ Relu | Sigmoid]                                          -> "convt"
  MaxPool, Concat(axis 1), Resize(nearest, integer scale), Add(tensor, tensor)       -> as is

GELU is the exact form Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5).
A LayerNorm (ReduceMean over the channel axis with Sub/Pow/Sqrt) is not in these graphs; any
ReduceMean that is not a squeeze-excite block is an error, so none can slip through.

Weights: one file, every blob 4-byte aligned, little-endian. A tensor has a kind (the shared packed spec):

  f32   n float32 values at byte offset "off"
  f16   n IEEE half values at "off"
  i8    n signed int8 codes at "off" in [-L, L], then one fp16 scale per output channel at "soff"
        (conv: weight dim 0; ConvTranspose: dim 1, its output-channel dim); dequantised value = code * scale

Biases and every tensor under 1,024 elements are f32 in every scheme. The scheme names the kind of the
rest: "f32", "f16", or "r<L>" (i8 with codes in +-L, L <= 127; scale = f16(max|row| / L), code =
rint(w / scale)). The manifest names each tensor's kind, so a loader needs nothing but the manifest.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("tiny", "small")
OUT_ROOT = ROOT / "web/det/models"
FORMAT = "ppocr-pack/det-program v1"
SCHEMES = ("r63", "r127", "f16", "f32")   # byte counts are reported for these
MIN_QUANT = 1024                          # tensors with fewer elements stay f32
SQRT2 = 1.4142135
TOL = 1e-5
ACT_CODE = {"none": 0, "relu": 1, "hsig": 2, "gelu": 3, "sigmoid": 4}


def attr(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def txt(v):
    return v.decode() if isinstance(v, bytes) else v


def parse_scheme(scheme):
    """"f32" | "f16" | "r<L>" (1 <= L <= 127) -> (kind, L)."""
    if scheme in ("f32", "f16"):
        return scheme, 0
    m = re.fullmatch(r"r(\d+)", scheme)
    if not m or not 1 <= int(m[1]) <= 127:
        raise SystemExit(f"scheme must be f32, f16 or r<L> with 1 <= L <= 127, got {scheme!r}")
    return "i8", int(m[1])


def quantize_rows(w, axis, L):
    """int8 codes in [-L, L] and one fp16 scale per slice along `axis` (the output channel).

    scale = f16(max|row| / L) computed in float32; code = rint(w / scale) clipped to +-L, so the
    dequantised weight is code * scale (exact in float32: 7 bits times an 11-bit significand).
    A row of zeros (or one whose scale underflows to 0 in fp16) gets codes 0.
    """
    a = np.moveaxis(np.asarray(w, dtype=np.float32), axis, 0)
    flat = a.reshape(a.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    scale = (amax / np.float32(L)).astype(np.float16)
    s32 = scale.astype(np.float32)
    safe = np.where(s32 > 0, s32, np.float32(1))
    codes = np.clip(np.rint(flat.astype(np.float64) / safe[:, None].astype(np.float64)), -L, L)
    codes = np.where(s32[:, None] > 0, codes, 0).astype(np.int8)
    codes = np.moveaxis(codes.reshape(a.shape), 0, axis)
    return np.ascontiguousarray(codes), scale


class Packer:
    """Appends 4-byte-aligned blobs to one buffer and describes each tensor for the manifest."""

    def __init__(self, scheme):
        self.scheme = scheme
        self.kind, self.L = parse_scheme(scheme)
        self.chunks, self.bytes = [], 0
        self.tally = defaultdict(lambda: [0, 0])   # kind -> [tensors, elements]

    def blob(self, raw):
        off = self.bytes
        pad = (-len(raw)) % 4
        self.chunks.append(raw + b"\0" * pad)
        self.bytes += len(raw) + pad
        return off

    def weight(self, arr, axis=0):
        """Weight tensor -> manifest dict {kind, off[, soff, L, axis]}; the kind follows the scheme."""
        a = np.asarray(arr, dtype=np.float32)
        kind = self.kind if a.size >= MIN_QUANT else "f32"
        self.tally[kind][0] += 1
        self.tally[kind][1] += a.size
        if kind == "f32":
            return {"kind": "f32", "off": self.blob(a.astype("<f4").tobytes())}
        if kind == "f16":
            return {"kind": "f16", "off": self.blob(a.astype("<f2").tobytes())}
        codes, scale = quantize_rows(a, axis, self.L)
        off = self.blob(codes.tobytes())
        return {"kind": "i8", "off": off, "soff": self.blob(scale.astype("<f2").tobytes()),
                "L": self.L, "axis": axis}

    def f32(self, arr):
        a = np.asarray(arr, dtype="<f4").ravel()
        return self.blob(a.tobytes())

    def data(self):
        return b"".join(self.chunks)


class Exporter:
    def __init__(self, model, scheme):
        self.g = model.graph
        self.init = {t.name: numpy_helper.to_array(t) for t in self.g.initializer}
        self.nodes = list(self.g.node)
        self.cons = defaultdict(list)
        for k, n in enumerate(self.nodes):
            for i in n.input:
                if i:
                    self.cons[i].append(k)
        self.graph_outputs = {o.name for o in self.g.output}
        self.absorbed, self.emitted = set(), set()
        self.buf, self.bufc, self.bufnames, self.ops = {}, [], [], []
        self.pack = Packer(scheme)
        self.stats = defaultdict(int)
        self.new_buf(self.g.input[0].name, 3)

    # ---------------------------------------------------------------- graph helpers
    def new_buf(self, name, c):
        self.buf[name] = len(self.bufc)
        self.bufc.append(c)
        self.bufnames.append(name)
        return self.buf[name]

    def sole(self, t):
        """Index of the only node consuming tensor t (and t is not a graph output), else None."""
        c = self.cons.get(t, [])
        return c[0] if len(c) == 1 and t not in self.graph_outputs else None

    def one(self, t, op_type, what):
        i = self.sole(t)
        if i is None or self.nodes[i].op_type != op_type:
            raise SystemExit(f"{what}: expected a single {op_type} after {t}")
        return i

    def is_const(self, name, value=None):
        v = self.init.get(name)
        if v is None:
            return False
        if value is None:
            return True
        return v.size == 1 and abs(float(v.reshape(-1)[0]) - value) < TOL

    # ---------------------------------------------------------------- activation tails
    def match_gelu(self, t):
        cs = self.cons.get(t, [])
        if len(cs) != 2 or t in self.graph_outputs:
            return None
        for di in cs:
            d = self.nodes[di]
            if d.op_type != "Div" or d.input[0] != t or not self.is_const(d.input[1], SQRT2):
                continue
            (m1i,) = [c for c in cs if c != di]
            m1 = self.nodes[m1i]
            if m1.op_type != "Mul" or set(m1.input) != {t, m1.input[0] if m1.input[0] != t else m1.input[1]}:
                continue
            ei = self.sole(d.output[0])
            if ei is None or self.nodes[ei].op_type != "Erf":
                continue
            ai = self.sole(self.nodes[ei].output[0])
            if ai is None or self.nodes[ai].op_type != "Add":
                continue
            a = self.nodes[ai]
            if not (self.is_const(a.input[0], 1.0) or self.is_const(a.input[1], 1.0)):
                continue
            if self.nodes[ai].output[0] not in m1.input:
                continue
            m2i = self.sole(m1.output[0])
            if m2i is None or self.nodes[m2i].op_type != "Mul":
                continue
            m2 = self.nodes[m2i]
            if not (self.is_const(m2.input[0], 0.5) or self.is_const(m2.input[1], 0.5)):
                continue
            return {"nodes": [di, ei, ai, m1i, m2i], "out": m2.output[0]}
        return None

    def tail(self, t):
        """Activation folded after a conv output t. Returns (act, output tensor, ha, hb)."""
        g = self.match_gelu(t)
        if g:
            self.absorbed.update(g["nodes"])
            self.stats["gelu"] += 1
            return "gelu", g["out"], 0.0, 0.0
        s = self.sole(t)
        if s is None:
            return "none", t, 0.0, 0.0
        m = self.nodes[s]
        if m.op_type == "Relu":
            act, ha, hb = "relu", 0.0, 0.0
        elif m.op_type == "HardSigmoid":
            a = attr(m)
            act, ha, hb = "hsig", float(a.get("alpha", 0.2)), float(a.get("beta", 0.5))
        elif m.op_type == "Sigmoid":
            act, ha, hb = "sigmoid", 0.0, 0.0
        else:
            return "none", t, 0.0, 0.0
        self.absorbed.add(s)
        self.stats[act] += 1
        return act, m.output[0], ha, hb

    def fold_bn(self, t, W, B):
        """If a BatchNormalization consumes t alone, fold it into W and B. Returns (t', W, B)."""
        s = self.sole(t)
        if s is None or self.nodes[s].op_type != "BatchNormalization":
            return t, W, B
        bn = self.nodes[s]
        gamma, beta, mean, var = [self.init[i].astype(np.float32) for i in bn.input[1:5]]
        f = gamma / np.sqrt(var + attr(bn).get("epsilon", 1e-5))
        self.absorbed.add(s)
        self.stats["batchnorm_folded"] += 1
        return bn.output[0], W * f[:, None, None, None], (B - mean) * f + beta

    # ---------------------------------------------------------------- geometry
    @staticmethod
    def pads_for(a, kernel, dil, stride):
        ap = txt(a.get("auto_pad", "NOTSET"))
        if ap == "VALID":
            return [0, 0, 0, 0]
        if ap in ("SAME_UPPER", "SAME_LOWER"):
            assert list(stride) == [1, 1], "SAME padding is exported for stride 1 only"
            th, tw = dil[0] * (kernel[0] - 1), dil[1] * (kernel[1] - 1)
            bh = (th + 1) // 2 if ap == "SAME_LOWER" else th // 2
            bw = (tw + 1) // 2 if ap == "SAME_LOWER" else tw // 2
            return [bh, bw, th - bh, tw - bw]
        assert ap == "NOTSET", ap
        return [int(v) for v in a.get("pads", [0, 0, 0, 0])]

    def conv_entry(self, W, B, act, ha, hb, geom, axis=0):
        """Manifest entry for a conv, ConvTranspose (axis 1) or squeeze-excite fc."""
        return dict(geom, act=act, ha=ha, hb=hb, wq=self.pack.weight(W, axis),
                    wshape=[int(v) for v in W.shape], b=self.pack.f32(B))

    def conv_geom(self, n, W):
        a = attr(n)
        k = [int(W.shape[2]), int(W.shape[3])]
        s = [int(v) for v in a.get("strides", [1, 1])]
        d = [int(v) for v in a.get("dilations", [1, 1])]
        return {"k": k, "s": s, "d": d, "pads": self.pads_for(a, k, d, s), "g": int(a.get("group", 1))}

    # ---------------------------------------------------------------- op handlers
    def op_Conv(self, k, n):
        W = self.init[n.input[1]].astype(np.float32)
        B = self.init[n.input[2]].astype(np.float32) if len(n.input) > 2 else np.zeros(W.shape[0], np.float32)
        geom = self.conv_geom(n, W)
        ib = self.buf[n.input[0]]
        t, W, B = n.output[0], W, B
        t, W, B = self.fold_bn(t, W, B)
        act, out_t, ha, hb = self.tail(t)
        e = self.conv_entry(W, B, act, ha, hb, geom)
        assert W.shape[1] * geom["g"] == self.bufc[ib], f"conv {n.name}: channels"
        self.ops.append(dict(op="conv", **{"in": ib}, out=self.new_buf(out_t, int(W.shape[0])), **e))
        self.emitted.add(k)

    def op_ConvTranspose(self, k, n):
        a = attr(n)
        W = self.init[n.input[1]].astype(np.float32)
        assert int(a.get("group", 1)) == 1 and list(a.get("kernel_shape")) == [2, 2]
        assert list(a.get("strides")) == [2, 2] and list(a.get("pads", [0, 0, 0, 0])) == [0, 0, 0, 0]
        B = np.zeros(W.shape[1], np.float32)
        t = n.output[0]
        s = self.sole(t)
        if s is not None and self.nodes[s].op_type == "Add":
            add = self.nodes[s]
            other = add.input[1] if add.input[0] == t else add.input[0]
            if self.is_const(other):
                B = self.init[other].astype(np.float32).reshape(-1)
                self.absorbed.add(s)
                t = add.output[0]
        act, out_t, ha, hb = self.tail(t)
        geom = {"k": [2, 2], "s": [2, 2], "d": [1, 1], "pads": [0, 0, 0, 0], "g": 1}
        e = self.conv_entry(W, B, act, ha, hb, geom, axis=1)
        self.ops.append(dict(op="convt", **{"in": self.buf[n.input[0]]},
                             out=self.new_buf(out_t, int(W.shape[1])), **e))
        self.emitted.add(k)

    def emit_se(self, k, n, x, pool_out):
        i1 = self.one(pool_out, "Conv", "SE fc1")
        r1 = self.one(self.nodes[i1].output[0], "Relu", "SE relu")
        i2 = self.one(self.nodes[r1].output[0], "Conv", "SE fc2")
        ih = self.one(self.nodes[i2].output[0], "HardSigmoid", "SE gate")
        hs = self.nodes[ih]
        im = self.sole(hs.output[0])
        mul = self.nodes[im] if im is not None else None
        if mul is None or mul.op_type != "Mul" or set(mul.input) != {x, hs.output[0]}:
            raise SystemExit(f"SE block at node {k}: gate does not multiply its input")
        chain, out_t, residual = [k, i1, r1, i2, ih, im], mul.output[0], False
        ia = self.sole(mul.output[0])
        if ia is not None and self.nodes[ia].op_type == "Add" and set(self.nodes[ia].input) == {x, mul.output[0]}:
            chain.append(ia)
            out_t, residual = self.nodes[ia].output[0], True
        assert set(self.cons[x]) == set(chain) - {i1, r1, i2, ih} | set(), "SE input has other consumers"
        for idx in chain[1:]:
            self.absorbed.add(idx)
        fc1, fc2 = self.nodes[i1], self.nodes[i2]
        W1 = self.init[fc1.input[1]].astype(np.float32)
        B1 = self.init[fc1.input[2]].astype(np.float32)
        W2 = self.init[fc2.input[1]].astype(np.float32)
        B2 = self.init[fc2.input[2]].astype(np.float32)
        one_by_one = {"k": [1, 1], "s": [1, 1], "d": [1, 1], "pads": [0, 0, 0, 0], "g": 1}
        a = attr(hs)
        bi = self.buf[x]
        self.ops.append(dict(
            op="se", **{"in": bi}, out=self.new_buf(out_t, self.bufc[bi]), res=residual,
            fc1=self.conv_entry(W1, B1, "relu", 0.0, 0.0, one_by_one),
            fc2=self.conv_entry(W2, B2, "hsig", float(a.get("alpha", 0.2)), float(a.get("beta", 0.5)), one_by_one)))
        self.stats["se"] += 1
        self.emitted.add(k)

    def op_ReduceMean(self, k, n):
        a = attr(n)
        if list(a.get("axes", [])) != [2, 3] or int(a.get("keepdims", 1)) != 1:
            raise SystemExit(f"ReduceMean at node {k} is not a squeeze-excite pool (axes [2,3])")
        self.emit_se(k, n, n.input[0], n.output[0])

    def op_GlobalAveragePool(self, k, n):
        self.emit_se(k, n, n.input[0], n.output[0])

    def op_MaxPool(self, k, n):
        a = attr(n)
        kern = [int(v) for v in a["kernel_shape"]]
        s = [int(v) for v in a.get("strides", [1, 1])]
        pads = self.pads_for(a, kern, [1, 1], s)
        ib = self.buf[n.input[0]]
        self.ops.append(dict(op="maxpool", **{"in": ib}, out=self.new_buf(n.output[0], self.bufc[ib]),
                             k=kern, s=s, d=[1, 1], pads=pads, g=1))
        self.emitted.add(k)

    def op_Concat(self, k, n):
        assert int(attr(n).get("axis", 1)) == 1
        ins = [self.buf[i] for i in n.input]
        c = sum(self.bufc[i] for i in ins)
        self.ops.append(dict(op="concat", ins=ins, out=self.new_buf(n.output[0], c)))
        self.emitted.add(k)

    def op_Resize(self, k, n):
        a = attr(n)
        assert txt(a.get("mode")) == "nearest"
        assert txt(a.get("coordinate_transformation_mode")) == "asymmetric"
        assert txt(a.get("nearest_mode")) == "floor"
        sc = self.init[n.input[2]].reshape(-1)
        assert sc[0] == 1 and sc[1] == 1 and sc[2] == sc[3] and float(sc[2]).is_integer(), sc
        ib = self.buf[n.input[0]]
        self.ops.append(dict(op="resize", **{"in": ib}, out=self.new_buf(n.output[0], self.bufc[ib]),
                             scale=int(sc[2])))
        self.emitted.add(k)

    def op_Add(self, k, n):
        if any(self.is_const(i) for i in n.input):
            raise SystemExit(f"Add with a constant at node {k} is not absorbed into a conv")
        a, b = (self.buf[i] for i in n.input)
        assert self.bufc[a] == self.bufc[b]
        self.ops.append(dict(op="add", ins=[a, b], out=self.new_buf(n.output[0], self.bufc[a])))
        self.emitted.add(k)

    # ---------------------------------------------------------------- driver
    def run(self):
        for k, n in enumerate(self.nodes):
            if k in self.absorbed:
                continue
            handler = getattr(self, "op_" + n.op_type, None)
            if handler is None:
                raise SystemExit(f"unsupported op {n.op_type} at node {k} ({n.name})")
            handler(k, n)
        assert not (self.absorbed & self.emitted)
        assert len(self.absorbed) + len(self.emitted) == len(self.nodes), "unhandled nodes"
        out = self.g.output[0].name
        assert self.bufc[self.buf[out]] == 1
        return out


def read_yaml(path):
    """(mean, std, DB postprocess settings) from the model's inference.yml."""
    cfg = yaml.safe_load(open(path))
    norm = next(op["NormalizeImage"] for op in cfg["PreProcess"]["transform_ops"] if "NormalizeImage" in op)
    assert norm.get("order") == "hwc", norm
    assert str(norm["scale"]).replace(" ", "") in ("1./255.", "1/255"), norm["scale"]
    pp = cfg["PostProcess"]
    assert pp["name"] == "DBPostProcess", pp
    post = {"thresh": float(pp["thresh"]), "box_thresh": float(pp["box_thresh"]),
            "unclip_ratio": float(pp["unclip_ratio"]), "max_candidates": int(pp["max_candidates"]),
            "min_size": 3, "score_mode": "fast"}
    return [float(v) for v in norm["mean"]], [float(v) for v in norm["std"]], post


def build(name, src_dir, scheme):
    """Walk the ONNX graph once. Returns (manifest dict, weights bytes, graph stats)."""
    model = onnx.load(str(src_dir / "inference.onnx"))
    ex = Exporter(model, scheme)
    out = ex.run()
    mean, std, post = read_yaml(src_dir / "inference.yml")
    data = ex.pack.data()
    assert len(data) % 4 == 0
    manifest = {
        "format": FORMAT,
        "model": name,
        "source": f"PP-OCRv6_{name}_det (PaddleOCR inference.onnx, Apache-2.0)",
        "scheme": scheme,
        "input": {"name": "x", "c": 3, "order": "BGR", "mean": mean, "std": std,
                  "scale": 1 / 255, "multiple": 32, "default_long_side": 1280},
        "postprocess": post,
        "buffers": [{"id": i, "c": c, "onnx": nm} for i, (c, nm) in enumerate(zip(ex.bufc, ex.bufnames))],
        "output": {"buf": ex.buf[out], "c": 1, "tensor": out},
        "weights": {"file": "weights.bin", "bytes": len(data)},
        "tensors": {k: {"count": v[0], "elements": v[1]} for k, v in sorted(ex.pack.tally.items())},
        "fusion": dict(ex.stats),
        "ops": ex.ops,
    }
    return manifest, data, ex, len(model.graph.node)


def gzip9(raw):
    return len(subprocess.run(["gzip", "-9", "-n", "-c"], input=raw, capture_output=True, check=True).stdout)


def manifest_bytes(manifest):
    return (json.dumps(manifest, separators=(",", ":")) + "\n").encode()


def export(name, src_dir, out_dir, scheme):
    manifest, data, ex, n_nodes = build(name, src_dir, scheme)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(data)
    (out_dir / "manifest.json").write_bytes(manifest_bytes(manifest))
    counts = defaultdict(int)
    for op in ex.ops:
        counts[op["op"]] += 1
    acts = defaultdict(int)
    for op in ex.ops:
        if op["op"] in ("conv", "convt"):
            acts[op["act"]] += 1
    print(f"{name} {scheme}: {n_nodes} onnx nodes -> {len(ex.ops)} ops {dict(counts)}; "
          f"conv/convt activations {dict(acts)}; se blocks {ex.stats['se']}; buffers {len(ex.bufc)}; "
          f"tensors {manifest['tensors']}; weights {len(data)} B -> {out_dir}")


def sizes(names, src, out_json):
    """Bytes of weights.bin and manifest.json for every scheme, raw and gzip -9."""
    res = {"note": "weights.bin and manifest.json as export writes them; gzip is `gzip -9 -n`",
           "models": {}}
    for name in names:
        res["models"][name] = {}
        for scheme in SCHEMES:
            manifest, data, _, _ = build(name, src / f"{name}_det", scheme)
            mb = manifest_bytes(manifest)
            res["models"][name][scheme] = {
                "weights_bytes": len(data), "weights_gzip9": gzip9(data),
                "manifest_bytes": len(mb), "manifest_gzip9": gzip9(mb),
                "total_bytes": len(data) + len(mb), "total_gzip9": gzip9(data) + gzip9(mb),
                "tensors": manifest["tensors"]}
            r = res["models"][name][scheme]
            print(f"{name:5s} {scheme:5s} weights {r['weights_bytes']:>9} B  gzip {r['weights_gzip9']:>9} B"
                  f"   manifest {r['manifest_bytes']:>6} B  gzip {r['manifest_gzip9']:>6} B")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1) + "\n")
    print("->", out_json)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", choices=NAMES, help="tiny and/or small (default both)")
    ap.add_argument("--scheme", default="r63", help="f32, f16 or r<L> with L <= 127 (default r63)")
    ap.add_argument("--src", default=os.environ.get("PPOCR_MODELS", str(ROOT / "models")),
                    help="folder holding <name>_det/inference.{onnx,yml}")
    ap.add_argument("--out", default=None, help="output root (default web/det/models); <out>/<name>/ is written")
    ap.add_argument("--sizes", action="store_true", help="write results/det_sizes.json for every scheme and stop")
    a = ap.parse_args(argv)
    names = a.names or list(NAMES)
    src = Path(a.src)
    if a.sizes:
        sizes(names, src, ROOT / "results/det_sizes.json")
        return
    out_root = Path(a.out) if a.out else OUT_ROOT
    for name in names:
        export(name, src / f"{name}_det", out_root / name, a.scheme)


if __name__ == "__main__":
    main(sys.argv[1:])
