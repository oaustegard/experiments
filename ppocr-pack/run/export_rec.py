#!/usr/bin/env python3
"""Export a PP-OCRv6 text-line recognizer (ONNX) to an op program plus packed weights.

    python3 -I run/export_rec.py                               # tiny and small, scheme r63
    python3 -I run/export_rec.py tiny --scheme r63,head=t4
    python3 -I run/export_rec.py small --scheme f32            # -> web/rec/models/small-f32/

Reads models/<name>_rec/inference.{onnx,yml} (run/fetch_models.py; --src for another directory) and
writes web/rec/models/<name>[-<scheme>]/{manifest.json, weights.bin, dict.txt}. run/ref_rec.py is the
NumPy executor of the same program and web/rec/engine.js the WGSL one.

The graph is walked once in topological order. Identity nodes are dropped and every node that only
computes a shape (Shape and whatever consumes only its results) is skipped. Every tensor lives in one
of two memory layouts, which is all the reshape plumbing in these graphs needs:

  chw  per line [C][H][W], channels at axis 1 (Conv outputs)
  tc   per line [T][C],    channels last     (sequence: LayerNorm, MatMul, attention)

Squeeze, Unsqueeze, Reshape and Flatten keep the row-major order, so they alias their input; a
Transpose that moves the channel axis becomes img2seq or seq2img (two that cancel are dropped).
Single-consumer chains fold into one op:

  Conv [+ Add(const bias)] [+ BatchNormalization] [+ Squeeze/Unsqueeze]
       [+ Relu | HardSigmoid | Sigmoid | exact GELU | hard-swish | swish]           -> conv
  ReduceMean(axes [2,3]) -> conv(+relu) -> conv(+hsig) -> Mul(x, gate) [-> Add(x)]   -> se
  ReduceMean(-1) Sub Pow ReduceMean Add(eps) Sqrt Div Mul(gamma) Add(beta)          -> ln
  MatMul(x, const W[K,N]) [+ Add(const bias)] [+ activation]                        -> linear
  Reshape [T,3,nh,hd] Transpose Slice*3 Mul(scale) MatMul Softmax MatMul Transpose Reshape -> attn
  AveragePool, MaxPool, Concat(axis 1), Add(tensor, tensor), Softmax(last axis)     -> as is

GELU is Div(x, sqrt2) Erf Add(1) Mul(x) Mul(0.5); hard-swish is x * HardSigmoid(x); swish is
x * Sigmoid(x). The linear op that feeds the final Softmax is the CTC head.

Packed weights (the scheme the sweep in run/quant.py simulates). BatchNorm and the bias Adds are folded
into the conv first. Per tensor format:

  f32   little-endian float32
  f16   one IEEE half per weight
  i8    int8 codes in [-L, L], one fp16 scale per output channel (conv: dim 0 of [O, I, kh, kw];
        MatMul weight [K, N]: per column N); scale = fp16(max|w| / L), code = round(w / scale)
  t4    4-bit codes in [-7, 7], groups of 32 consecutive K values within a column of [K, N], one fp16
        scale per group (max|w| / 7); stored column-major [N][K padded to 32], code + 8 in a nibble,
        low nibble first. CTC head only.

Weights under 1,024 values and every bias and LayerNorm parameter stay f32. The dequantised value is
float32(code) * float32(scale), rounded once to f32, in the exporter, run/ref_rec.py and engine.js.

Scheme grammar: comma-separated items, later ones win. A bare format applies to every part;
`part=format` to one. Parts: conv (all conv weights, squeeze-excite included), mm (MatMul weights
other than the head), head (the CTC head), all. Formats: f32, f16, r<L> (i8 with codes in [-L, L],
L <= 127), t4 (alias g32r7; head only). Default r63.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models"
OUT_ROOT = ROOT / "web/rec/models"
FORMAT = "ppocr-pack/rec-program v1"
NAMES = ("tiny", "small")
DEFAULT_SCHEME = "r63"
PARTS = ("conv", "mm", "head")
MIN_QUANT = 1024
SQRT2 = 1.4142135
TOL = 1e-5
PROBE_W = 328   # probe width: makes T = 41, a size no channel count shares
ALIAS_OPS = ("Squeeze", "Unsqueeze", "Reshape", "Flatten")


def attr(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def txt(v):
    return v.decode() if isinstance(v, bytes) else v


# ---------------------------------------------------------------------------------------- scheme


def parse_scheme(scheme):
    spec = {p: "f32" for p in PARTS}
    for item in scheme.split(","):
        item = item.strip()
        part, eq, fmt = item.partition("=")
        if not eq:
            parts, fmt = PARTS, part
        else:
            parts = PARTS if part == "all" else (part,)
        if fmt == "g32r7":
            fmt = "t4"
        m = re.fullmatch(r"r(\d+)", fmt)
        if not (fmt in ("f32", "f16", "t4") or (m and 1 <= int(m[1]) <= 127)):
            raise SystemExit(f"bad format {fmt!r} in scheme {scheme!r}")
        for p in parts:
            if p not in spec:
                raise SystemExit(f"unknown part {p!r} in scheme {scheme!r}")
            if fmt == "t4" and p != "head":
                if eq:
                    raise SystemExit("t4 is for the CTC head only")
                continue   # a bare t4 means the head
            spec[p] = fmt
    return spec


def f16_scale(amax, L):
    """fp16 scale per row and its f32 value (0 for an all-zero row)."""
    s = (np.asarray(amax, np.float32) / np.float32(L)).astype(np.float16)
    return s, s.astype(np.float32)


def quant_rows(w2, L):
    """w2 [R, M] float32 -> int8 codes [R, M], fp16 scales [R]; one scale per row."""
    s16, s = f16_scale(np.abs(w2).max(axis=1), L)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s[:, None] > 0, np.rint(w2 / s[:, None]), 0)
    return np.clip(q, -L, L).astype(np.int8), s16


def quant_t4(w, G=32):
    """w [K, N] -> codes [N, Kp] int8 in [-7, 7] (Kp = K rounded up to G), fp16 scales [N, Kp/G]."""
    K, N = w.shape
    Kp = -(-K // G) * G
    wt = np.zeros((N, Kp), np.float32)
    wt[:, :K] = w.T
    g = wt.reshape(N, Kp // G, G)
    s16, s = f16_scale(np.abs(g).max(axis=2), 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(s[..., None] > 0, np.rint(g / s[..., None]), 0)
    return np.clip(q, -7, 7).astype(np.int8).reshape(N, Kp), s16


def dequant(e, raw):
    """The f32 values of manifest tensor entry e from weights.bin bytes (shared with ref_rec.py)."""
    shape, n = e["shape"], int(np.prod(e["shape"]))
    if e["fmt"] == "f32":
        return np.frombuffer(raw, "<f4", n, e["off"]).reshape(shape).copy()
    if e["fmt"] == "f16":
        return np.frombuffer(raw, "<f2", n, e["off"]).astype(np.float32).reshape(shape)
    if e["fmt"] == "i8":
        q = np.frombuffer(raw, np.int8, n, e["off"]).astype(np.float32)
        s = np.frombuffer(raw, "<f2", e["nsc"], e["sc"]).astype(np.float32)
        if e["axis"] == 0:
            return (q.reshape(e["nsc"], -1) * s[:, None]).reshape(shape)
        return (q.reshape(-1, e["nsc"]) * s[None, :]).reshape(shape)
    if e["fmt"] == "t4":
        K, N = shape
        G = e["group"]
        Kp = -(-K // G) * G
        b = np.frombuffer(raw, np.uint8, N * Kp // 2, e["off"])
        q = np.empty(N * Kp, np.float32)
        q[0::2] = (b & 15).astype(np.float32) - 8
        q[1::2] = (b >> 4).astype(np.float32) - 8
        s = np.frombuffer(raw, "<f2", N * Kp // G, e["sc"]).astype(np.float32)
        w = (q.reshape(N, Kp // G, G) * s.reshape(N, Kp // G, 1)).reshape(N, Kp)
        return np.ascontiguousarray(w[:, :K].T)
    raise ValueError(e["fmt"])


class Blob:
    def __init__(self):
        self.chunks, self.size = [], 0

    def add(self, b):
        off = self.size
        pad = (-len(b)) % 4
        self.chunks.append(b + b"\0" * pad)
        self.size += len(b) + pad
        return off

    def data(self):
        return b"".join(self.chunks)


def pack(tensors, spec):
    """Pack the exporter's tensors per scheme spec. Returns (bytes, manifest tensor entries, stats)."""
    blob = Blob()
    entries = []
    stats = defaultdict(lambda: {"tensors": 0, "values": 0, "bytes": 0})
    for i, t in enumerate(tensors):
        a = np.ascontiguousarray(t["data"], np.float32)
        fmt = "f32"
        if t["role"] in PARTS and a.size >= MIN_QUANT:
            fmt = spec[t["role"]]
        e = {"id": i, "name": t["name"], "role": t["role"], "shape": [int(v) for v in a.shape]}
        start = blob.size
        if fmt == "f32":
            e.update(fmt="f32", off=blob.add(a.astype("<f4").tobytes()))
        elif fmt == "f16":
            e.update(fmt="f16", off=blob.add(a.astype("<f2").tobytes()))
        elif fmt == "t4":
            assert a.ndim == 2, "t4 is for a MatMul weight [K, N]"
            q, s16 = quant_t4(a)
            u = (q.reshape(-1) + 8).astype(np.uint8)
            e.update(fmt="t4", group=32, off=blob.add((u[0::2] | (u[1::2] << 4)).astype(np.uint8).tobytes()),
                     sc=blob.add(s16.astype("<f2").tobytes()), nsc=int(s16.size))
        else:
            L = int(fmt[1:])
            if t["role"] == "conv":
                q, s16 = quant_rows(a.reshape(a.shape[0], -1), L)
                axis = 0
            else:   # MatMul [K, N]: one scale per column
                qt, s16 = quant_rows(np.ascontiguousarray(a.T), L)
                q, axis = qt.T, 1
            e.update(fmt="i8", L=L, axis=axis, off=blob.add(np.ascontiguousarray(q).astype(np.int8).tobytes()),
                     sc=blob.add(s16.astype("<f2").tobytes()), nsc=int(s16.size))
        e["bytes"] = blob.size - start
        st = stats[e["fmt"] if e["fmt"] != "i8" else f"i8(r{e['L']})"]
        st["tensors"] += 1
        st["values"] += int(a.size)
        st["bytes"] += e["bytes"]
        entries.append(e)
    return blob.data(), entries, dict(stats)


# ---------------------------------------------------------------------------------------- probe


def probe_shapes(model, W=PROBE_W):
    """Concrete shape of every tensor at input [1, 3, 48, W], from onnxruntime with all outputs exposed."""
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(model)
    inferred = onnx.shape_inference.infer_shapes(m)
    types = {vi.name: vi.type.tensor_type.elem_type for vi in
             list(inferred.graph.value_info) + list(inferred.graph.output)}
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have and types.get(o):
                m.graph.output.append(helper.make_tensor_value_info(o, types[o], None))
                have.add(o)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    x = np.zeros((1, 3, 48, W), np.float32)
    outs = sess.run(names, {model.graph.input[0].name: x})
    shapes = {nm: list(np.asarray(v).shape) for nm, v in zip(names, outs)}
    shapes[model.graph.input[0].name] = [1, 3, 48, W]
    return shapes


# ---------------------------------------------------------------------------------------- exporter


class Exporter:
    def __init__(self, model, shapes):
        g = model.graph
        self.init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        self.shapes = shapes
        alias = {}

        def canon(t):
            while t in alias:
                t = alias[t]
            return t

        self.nodes = []
        for n in g.node:
            if n.op_type == "Identity":
                alias[n.output[0]] = canon(n.input[0])
                continue
            for i, nm in enumerate(n.input):
                if nm:
                    n.input[i] = canon(nm)
            self.nodes.append(n)
        self.n_identity = len(g.node) - len(self.nodes)
        self.graph_out = canon(g.output[0].name)
        self.graph_out_name = g.output[0].name
        # shape-only nodes: Shape, and nodes all of whose non-constant inputs are shape-only
        self.shape_only, self.skipped = set(), set()
        for k, n in enumerate(self.nodes):
            data = [i for i in n.input if i and i not in self.init]
            if n.op_type == "Shape" or (data and all(i in self.shape_only for i in data)):
                self.shape_only.update(n.output)
                self.skipped.add(k)
        self.cons = defaultdict(list)
        for k, n in enumerate(self.nodes):
            if k in self.skipped:
                continue
            ins = n.input[:1] if n.op_type == "Reshape" else n.input
            for i in ins:
                if i and i not in self.init:
                    self.cons[i].append(k)
        self.absorbed, self.emitted = set(), set()
        self.buf, self.bufc, self.bufkind, self.bufnames, self.ops = {}, [], [], [], []
        self.tensors = []
        self.stats = defaultdict(int)
        self.new_buf(g.input[0].name, 3, "chw")

    # ------------------------------------------------------------------ helpers
    def new_buf(self, name, c, kind):
        self.buf[name] = len(self.bufc)
        self.bufc.append(int(c))
        self.bufkind.append(kind)
        self.bufnames.append(name)
        return self.buf[name]

    def tensor(self, arr, role, name):
        self.tensors.append({"data": np.asarray(arr, np.float32), "role": role, "name": name})
        return len(self.tensors) - 1

    def sole(self, t):
        c = self.cons.get(t, [])
        return c[0] if len(c) == 1 and t != self.graph_out else None

    def one(self, t, op_type, what):
        i = self.sole(t)
        if i is None or self.nodes[i].op_type != op_type:
            raise SystemExit(f"{what}: expected a single {op_type} after {t}")
        return i

    def const(self, name, value=None):
        v = self.init.get(name)
        if v is None:
            return None
        if value is not None and not (v.size == 1 and abs(float(v.reshape(-1)[0]) - value) < TOL):
            return None
        return v

    def other_const(self, node, t):
        """For a binary node with input t, the constant on the other side (or None)."""
        other = node.input[1] if node.input[0] == t else node.input[0]
        return self.const(other)

    def axes(self, n):
        a = attr(n).get("axes")
        if a is None and len(n.input) > 1:
            a = self.init[n.input[1]].tolist()
        return [int(v) for v in a]

    # ------------------------------------------------------------------ activation tails
    def match_gelu(self, t):
        cs = self.cons.get(t, [])
        if len(cs) != 2 or t == self.graph_out:
            return None
        for di in cs:
            d = self.nodes[di]
            if d.op_type != "Div" or d.input[0] != t or self.const(d.input[1], SQRT2) is None:
                continue
            (m1i,) = [c for c in cs if c != di]
            m1 = self.nodes[m1i]
            if m1.op_type != "Mul" or t not in m1.input:
                continue
            ei = self.sole(d.output[0])
            if ei is None or self.nodes[ei].op_type != "Erf":
                continue
            ai = self.sole(self.nodes[ei].output[0])
            if ai is None or self.nodes[ai].op_type != "Add":
                continue
            a = self.nodes[ai]
            if self.const(a.input[0], 1.0) is None and self.const(a.input[1], 1.0) is None:
                continue
            if a.output[0] not in m1.input:
                continue
            m2i = self.sole(m1.output[0])
            if m2i is None or self.nodes[m2i].op_type != "Mul":
                continue
            m2 = self.nodes[m2i]
            if self.const(m2.input[0], 0.5) is None and self.const(m2.input[1], 0.5) is None:
                continue
            return [di, ei, ai, m1i, m2i], m2.output[0]
        return None

    def match_gated(self, t, gate_op):
        """x * gate(x): t has two consumers, gate_op(t) and Mul(t, gate). Returns (nodes, out, gate node)."""
        cs = self.cons.get(t, [])
        if len(cs) != 2 or t == self.graph_out:
            return None
        for gi in cs:
            gn = self.nodes[gi]
            if gn.op_type != gate_op or gn.input[0] != t:
                continue
            (mi,) = [c for c in cs if c != gi]
            m = self.nodes[mi]
            if m.op_type == "Mul" and set(m.input) == {t, gn.output[0]} and self.sole(gn.output[0]) == mi:
                return [gi, mi], m.output[0], gn
        return None

    def tail(self, t):
        """Activation folded after t. Returns (act, output tensor, ha, hb)."""
        g = self.match_gelu(t)
        if g:
            self.absorbed.update(g[0])
            self.stats["act_gelu"] += 1
            return "gelu", g[1], 0.0, 0.0
        g = self.match_gated(t, "HardSigmoid")
        if g:
            a = attr(g[2])
            self.absorbed.update(g[0])
            self.stats["act_hswish"] += 1
            return "hswish", g[1], float(a.get("alpha", 0.2)), float(a.get("beta", 0.5))
        g = self.match_gated(t, "Sigmoid")
        if g:
            self.absorbed.update(g[0])
            self.stats["act_swish"] += 1
            return "swish", g[1], 0.0, 0.0
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
        self.stats["act_" + act] += 1
        return act, m.output[0], ha, hb

    def squeeze_keeps_chw(self, m):
        """Squeeze/Unsqueeze of a unit H axis (axis 2) of a channels-at-axis-1 tensor."""
        if m.op_type not in ("Squeeze", "Unsqueeze") or self.axes(m) != [2]:
            return False
        si, so = self.shapes[m.input[0]], self.shapes[m.output[0]]
        return (m.op_type == "Squeeze" and len(si) == 4 and si[2] == 1) or (m.op_type == "Unsqueeze" and len(si) == 3)

    def fold_conv_tail(self, t, W, B):
        """Fold constant bias Adds, BatchNorms and unit-axis squeezes after a conv output t."""
        while True:
            s = self.sole(t)
            if s is None:
                break
            m = self.nodes[s]
            if m.op_type == "Add":
                c = self.other_const(m, t)
                if c is None or c.size != W.shape[0]:
                    break
                B = B + c.astype(np.float32).reshape(-1)
                self.stats["bias_add_folded"] += 1
            elif m.op_type == "BatchNormalization":
                gamma, beta, mean, var = [self.init[i].astype(np.float32) for i in m.input[1:5]]
                f = gamma / np.sqrt(var + np.float32(attr(m).get("epsilon", 1e-5)))
                W = W * f.reshape(-1, *([1] * (W.ndim - 1)))
                B = (B - mean) * f + beta
                self.stats["batchnorm_folded"] += 1
            elif self.squeeze_keeps_chw(m):
                pass
            else:
                break
            self.absorbed.add(s)
            t = m.output[0]
        act, t, ha, hb = self.tail(t)
        return t, W.astype(np.float32), B.astype(np.float32), act, ha, hb

    # ------------------------------------------------------------------ geometry
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

    def conv_geom(self, n, W):
        a = attr(n)
        k = [int(W.shape[2]), int(W.shape[3])]
        s = [int(v) for v in a.get("strides", [1, 1])]
        d = [int(v) for v in a.get("dilations", [1, 1])]
        return {"k": k, "s": s, "d": d, "pads": self.pads_for(a, k, d, s), "g": int(a.get("group", 1))}

    def conv_entry(self, W, B, act, ha, hb, geom, name):
        return dict(geom, act=act, ha=ha, hb=hb, w=self.tensor(W, "conv", name), b=self.tensor(B, "bias", name + ".bias"),
                    wshape=[int(v) for v in W.shape])

    def chw_in(self, t, what):
        bi = self.buf[t]
        if self.bufkind[bi] != "chw":
            raise SystemExit(f"{what}: input {t} is not in channel-first layout")
        return bi

    def tc_in(self, t, what):
        bi = self.buf[t]
        if self.bufkind[bi] != "tc":
            raise SystemExit(f"{what}: input {t} is not a sequence [T, C]")
        return bi

    # ------------------------------------------------------------------ op handlers
    def op_Conv(self, k, n):
        W = self.init[n.input[1]].astype(np.float32)
        B = self.init[n.input[2]].astype(np.float32) if len(n.input) > 2 and n.input[2] else np.zeros(W.shape[0], np.float32)
        geom = self.conv_geom(n, W)
        ib = self.chw_in(n.input[0], f"conv {n.name}")
        assert W.shape[1] * geom["g"] == self.bufc[ib], f"conv {n.name}: channels"
        out_t, W, B, act, ha, hb = self.fold_conv_tail(n.output[0], W, B)
        e = self.conv_entry(W, B, act, ha, hb, geom, n.input[1])
        self.ops.append(dict(op="conv", **{"in": ib}, out=self.new_buf(out_t, W.shape[0], "chw"), **e))
        self.emitted.add(k)

    def emit_se(self, k, n):
        x, pool_out = n.input[0], n.output[0]
        convs = []
        t = pool_out
        for want in ("relu", "hsig"):
            i = self.one(t, "Conv", "SE fc")
            c = self.nodes[i]
            W = self.init[c.input[1]].astype(np.float32)
            B = self.init[c.input[2]].astype(np.float32) if len(c.input) > 2 and c.input[2] else np.zeros(W.shape[0], np.float32)
            geom = self.conv_geom(c, W)
            assert geom["k"] == [1, 1] and geom["g"] == 1
            self.absorbed.add(i)
            t, W, B, act, ha, hb = self.fold_conv_tail(c.output[0], W, B)
            if act != want:
                raise SystemExit(f"SE block at node {k}: fc activation {act}, expected {want}")
            convs.append(self.conv_entry(W, B, act, ha, hb, geom, c.input[1]))
        im = self.sole(t)
        mul = self.nodes[im] if im is not None else None
        if mul is None or mul.op_type != "Mul" or set(mul.input) != {x, t}:
            raise SystemExit(f"SE block at node {k}: gate does not multiply its input")
        self.absorbed.add(im)
        out_t, residual = mul.output[0], False
        ia = self.sole(mul.output[0])
        if ia is not None and self.nodes[ia].op_type == "Add" and set(self.nodes[ia].input) == {x, mul.output[0]}:
            self.absorbed.add(ia)
            out_t, residual = self.nodes[ia].output[0], True
        expect = {k, im} | ({ia} if residual else set())
        if set(self.cons[x]) != expect:
            raise SystemExit(f"SE block at node {k}: its input has other consumers")
        bi = self.chw_in(x, "SE")
        self.ops.append(dict(op="se", **{"in": bi}, out=self.new_buf(out_t, self.bufc[bi], "chw"), res=residual,
                             fc1=convs[0], fc2=convs[1]))
        self.stats["se"] += 1
        self.emitted.add(k)

    def emit_ln(self, k, n):
        x, mean = n.input[0], n.output[0]
        bi = self.tc_in(x, "LayerNorm")
        D = self.bufc[bi]
        isub = self.one(mean, "Sub", "LayerNorm")
        sub = self.nodes[isub]
        assert list(sub.input) == [x, mean], "LayerNorm: Sub(x, mean)"
        d = sub.output[0]
        dc = self.cons[d]
        pw = [i for i in dc if self.nodes[i].op_type == "Pow"]
        dv = [i for i in dc if self.nodes[i].op_type == "Div"]
        assert len(dc) == 2 and len(pw) == 1 and len(dv) == 1, "LayerNorm: x - mean feeds Pow and Div"
        ip, idv = pw[0], dv[0]
        assert self.const(self.nodes[ip].input[1], 2.0) is not None
        ir = self.one(self.nodes[ip].output[0], "ReduceMean", "LayerNorm variance")
        assert self.axes(self.nodes[ir]) in ([-1], [len(self.shapes[x]) - 1])
        ia = self.one(self.nodes[ir].output[0], "Add", "LayerNorm eps")
        eps = float(self.other_const(self.nodes[ia], self.nodes[ir].output[0]).reshape(-1)[0])
        iq = self.one(self.nodes[ia].output[0], "Sqrt", "LayerNorm")
        assert self.sole(self.nodes[iq].output[0]) == idv and list(self.nodes[idv].input) == [d, self.nodes[iq].output[0]]
        im = self.one(self.nodes[idv].output[0], "Mul", "LayerNorm gamma")
        gamma = self.other_const(self.nodes[im], self.nodes[idv].output[0])
        ib = self.one(self.nodes[im].output[0], "Add", "LayerNorm beta")
        beta = self.other_const(self.nodes[ib], self.nodes[im].output[0])
        assert gamma is not None and beta is not None and gamma.size == D and beta.size == D
        self.absorbed.update([isub, ip, ir, ia, iq, idv, im, ib])
        out_t = self.nodes[ib].output[0]
        self.ops.append(dict(op="ln", **{"in": bi}, out=self.new_buf(out_t, D, "tc"), D=D, eps=eps,
                             g=self.tensor(gamma.reshape(-1), "norm", n.name + ".gamma"),
                             b=self.tensor(beta.reshape(-1), "norm", n.name + ".beta")))
        self.stats["ln"] += 1
        self.emitted.add(k)

    def op_ReduceMean(self, k, n):
        ax = self.axes(n)
        if ax == [2, 3] and int(attr(n).get("keepdims", 1)) == 1:
            return self.emit_se(k, n)
        if ax in ([-1], [len(self.shapes[n.input[0]]) - 1]) and self.bufkind[self.buf[n.input[0]]] == "tc":
            return self.emit_ln(k, n)
        raise SystemExit(f"ReduceMean at node {k} is neither squeeze-excite nor LayerNorm")

    def op_MatMul(self, k, n):
        if n.input[1] not in self.init:
            raise SystemExit(f"MatMul at node {k} has no constant weight and is not inside an attention block")
        W = self.init[n.input[1]].astype(np.float32)
        bi = self.tc_in(n.input[0], f"MatMul {n.name}")
        K, N = W.shape
        assert self.bufc[bi] == K, f"MatMul {n.name}: {self.bufc[bi]} inputs, weight {W.shape}"
        t, B = n.output[0], np.zeros(N, np.float32)
        s = self.sole(t)
        if s is not None and self.nodes[s].op_type == "Add":
            c = self.other_const(self.nodes[s], t)
            if c is not None and c.size == N:
                B = c.astype(np.float32).reshape(-1)
                self.absorbed.add(s)
                t = self.nodes[s].output[0]
        act, t, ha, hb = self.tail(t)
        self.ops.append(dict(op="linear", **{"in": bi}, out=self.new_buf(t, N, "tc"), K=int(K), N=int(N),
                             act=act, ha=ha, hb=hb, head=False,
                             w=self.tensor(W, "mm", n.input[1]), b=self.tensor(B, "bias", n.input[1] + ".bias")))
        self.emitted.add(k)

    def emit_attention(self, k, n):
        qkv = n.input[0]
        bi = self.tc_in(qkv, "attention")
        shp = self.shapes[n.output[0]]
        assert len(shp) == 5 and shp[2] == 3 and 3 * shp[3] * shp[4] == self.bufc[bi], shp
        nh, hd = int(shp[3]), int(shp[4])
        used = []
        it = self.one(n.output[0], "Transpose", "attention qkv")
        assert list(attr(self.nodes[it])["perm"]) == [2, 0, 3, 1, 4]
        used.append(it)
        parts = {}
        for ci in self.cons[self.nodes[it].output[0]]:
            c = self.nodes[ci]
            assert c.op_type == "Slice", "attention: q, k, v are slices"
            st, en = int(self.init[c.input[1]][0]), int(self.init[c.input[2]][0])
            ax = int(self.init[c.input[3]][0]) if len(c.input) > 3 else 0
            assert ax == 0 and en == st + 1
            isq = self.one(c.output[0], "Squeeze", "attention slice")
            parts[st] = self.nodes[isq].output[0]
            used += [ci, isq]
        q, kk, v = parts[0], parts[1], parts[2]
        im = self.one(q, "Mul", "attention scale")
        scale = float(self.other_const(self.nodes[im], q).reshape(-1)[0])
        ir = self.one(self.nodes[im].output[0], "Reshape", "attention scale")
        qs = self.nodes[ir].output[0]
        ikt = self.one(kk, "Transpose", "attention k")
        assert list(attr(self.nodes[ikt])["perm"]) == [0, 1, 3, 2]
        kt = self.nodes[ikt].output[0]
        imm = self.one(qs, "MatMul", "attention scores")
        assert list(self.nodes[imm].input) == [qs, kt] and self.sole(kt) == imm
        isf = self.one(self.nodes[imm].output[0], "Softmax", "attention")
        assert int(attr(self.nodes[isf]).get("axis", -1)) in (3, -1)
        imm2 = self.one(self.nodes[isf].output[0], "MatMul", "attention mix")
        assert list(self.nodes[imm2].input) == [self.nodes[isf].output[0], v] and self.sole(v) == imm2
        it2 = self.one(self.nodes[imm2].output[0], "Transpose", "attention out")
        assert list(attr(self.nodes[it2])["perm"]) == [0, 2, 1, 3]
        ir2 = self.one(self.nodes[it2].output[0], "Reshape", "attention out")
        out_t = self.nodes[ir2].output[0]
        assert self.shapes[out_t][-1] == nh * hd and len(self.shapes[out_t]) == 3
        used += [im, ir, ikt, imm, isf, imm2, it2, ir2]
        self.absorbed.update(used)
        self.ops.append(dict(op="attn", **{"in": bi}, out=self.new_buf(out_t, nh * hd, "tc"), nh=nh, hd=hd, scale=scale))
        self.stats["attn"] += 1
        self.emitted.add(k)

    def op_alias(self, k, n):
        x, y = n.input[0], n.output[0]
        bi = self.buf[x]
        kind, c = self.bufkind[bi], self.bufc[bi]
        si, so = self.shapes[x], self.shapes[y]
        if n.op_type == "Reshape" and kind == "tc" and len(so) == 5:
            return self.emit_attention(k, n)
        if kind == "chw":
            ok = len(so) >= 3 and so[1] == c and (len(si) < 4 or si[2] == 1) and (len(so) < 4 or so[2] == 1)
        else:
            ok = so[-1] == c and int(np.prod(so[1:-1])) == int(np.prod(si[1:-1]))
        if not ok:
            raise SystemExit(f"{n.op_type} at node {k} ({si} -> {so}) is not a layout-preserving alias of a {kind} tensor")
        self.buf[y] = bi
        self.stats["alias"] += 1
        self.emitted.add(k)

    op_Squeeze = op_Unsqueeze = op_Reshape = op_Flatten = op_alias

    def op_Transpose(self, k, n):
        perm = [int(v) for v in attr(n)["perm"]]
        x, y = n.input[0], n.output[0]
        bi = self.buf[x]
        kind = self.bufkind[bi]
        r = len(self.shapes[x])
        s = self.sole(y)
        if s is not None and self.nodes[s].op_type == "Transpose":
            p2 = [int(v) for v in attr(self.nodes[s])["perm"]]
            if [perm[i] for i in p2] == list(range(r)):
                self.absorbed.add(s)
                self.buf[self.nodes[s].output[0]] = bi
                self.stats["transpose_pair_dropped"] += 1
                self.emitted.add(k)
                return
        assert perm[0] == 0, f"Transpose at node {k} moves the batch axis"
        ch = 1 if kind == "chw" else r - 1
        new_ch = perm.index(ch)
        rest = [p for p in perm[1:] if p != ch]
        if rest != sorted(rest):
            raise SystemExit(f"Transpose at node {k} reorders spatial axes: {perm}")
        if kind == "chw" and new_ch == r - 1:
            self.ops.append(dict(op="img2seq", **{"in": bi}, out=self.new_buf(y, self.bufc[bi], "tc")))
        elif kind == "tc" and new_ch == 1:
            self.ops.append(dict(op="seq2img", **{"in": bi}, out=self.new_buf(y, self.bufc[bi], "chw")))
        else:
            raise SystemExit(f"Transpose at node {k} ({kind}, perm {perm}) is not a layout change")
        self.emitted.add(k)

    def op_Softmax(self, k, n):
        x = n.input[0]
        bi = self.tc_in(x, "Softmax")
        ax = int(attr(n).get("axis", -1))
        assert ax in (-1, len(self.shapes[x]) - 1), "Softmax over the last axis only"
        self.ops.append(dict(op="softmax", **{"in": bi}, out=self.new_buf(n.output[0], self.bufc[bi], "tc")))
        self.emitted.add(k)

    def op_AveragePool(self, k, n):
        a = attr(n)
        kern = [int(v) for v in a["kernel_shape"]]
        s = [int(v) for v in a.get("strides", [1, 1])]
        assert int(a.get("ceil_mode", 0)) == 0
        pads = self.pads_for(a, kern, [1, 1], s)
        ib = self.chw_in(n.input[0], "AveragePool")
        self.ops.append(dict(op="avgpool", **{"in": ib}, out=self.new_buf(n.output[0], self.bufc[ib], "chw"),
                             k=kern, s=s, pads=pads, cip=int(a.get("count_include_pad", 0))))
        self.emitted.add(k)

    def op_MaxPool(self, k, n):
        a = attr(n)
        kern = [int(v) for v in a["kernel_shape"]]
        s = [int(v) for v in a.get("strides", [1, 1])]
        assert int(a.get("ceil_mode", 0)) == 0
        pads = self.pads_for(a, kern, [1, 1], s)
        ib = self.chw_in(n.input[0], "MaxPool")
        self.ops.append(dict(op="maxpool", **{"in": ib}, out=self.new_buf(n.output[0], self.bufc[ib], "chw"),
                             k=kern, s=s, pads=pads))
        self.emitted.add(k)

    def op_Concat(self, k, n):
        assert int(attr(n).get("axis", 1)) == 1
        ins = [self.chw_in(i, "Concat") for i in n.input]
        self.ops.append(dict(op="concat", ins=ins, out=self.new_buf(n.output[0], sum(self.bufc[i] for i in ins), "chw")))
        self.emitted.add(k)

    def op_Add(self, k, n):
        if any(i in self.init for i in n.input):
            raise SystemExit(f"Add with a constant at node {k} was not folded into a conv or linear")
        a, b = (self.buf[i] for i in n.input)
        assert self.bufc[a] == self.bufc[b] and self.bufkind[a] == self.bufkind[b], f"Add at node {k}"
        self.ops.append(dict(op="add", ins=[a, b], out=self.new_buf(n.output[0], self.bufc[a], self.bufkind[a])))
        self.emitted.add(k)

    # ------------------------------------------------------------------ driver
    def run(self):
        for k, n in enumerate(self.nodes):
            if k in self.absorbed or k in self.skipped:
                continue
            handler = getattr(self, "op_" + n.op_type, None)
            if handler is None:
                raise SystemExit(f"unsupported op {n.op_type} at node {k} ({n.name})")
            handler(k, n)
        assert not (self.absorbed & self.emitted)
        assert len(self.absorbed) + len(self.emitted) + len(self.skipped) == len(self.nodes), "unhandled nodes"
        out = self.buf[self.graph_out]
        last = self.ops[-1]
        assert last["op"] == "softmax" and last["out"] == out, "the program must end in the CTC softmax"
        head = [op for op in self.ops if op["op"] == "linear" and op["out"] == last["in"]]
        assert len(head) == 1, "the softmax input must come from a linear head"
        head[0]["head"] = True
        self.tensors[head[0]["w"]]["role"] = "head"
        return out


# ---------------------------------------------------------------------------------------- driver


def dictionary(yml):
    cfg = yaml.safe_load(open(yml, encoding="utf-8"))
    post = cfg["PostProcess"]
    assert post["name"] == "CTCLabelDecode", post["name"]
    chars = [str(c) for c in post["character_dict"]]
    assert all(c and "\n" not in c and "\r" not in c for c in chars), "dictionary entry with a line break"
    rr = next((op["RecResizeImg"] for op in cfg["PreProcess"]["transform_ops"] if "RecResizeImg" in op), None)
    return chars, (rr or {}).get("image_shape", [3, 48, 320])


def slug(scheme):
    return scheme.replace(",", "_").replace("=", "-")


def export(name, src, out_dir, scheme):
    model = onnx.load(str(src / "inference.onnx"))
    n_nodes = len(model.graph.node)
    shapes = probe_shapes(model)
    ex = Exporter(model, shapes)
    out_buf = ex.run()
    chars, image_shape = dictionary(src / "inference.yml")
    V = ex.bufc[out_buf]
    assert V == len(chars) + 2, f"V {V} != {len(chars)} dictionary entries + blank + space"
    spec = parse_scheme(scheme)
    data, entries, wstats = pack(ex.tensors, spec)
    # the packed values must dequantise to within one step of the folded fp32 weights
    for e, t in zip(entries, ex.tensors):
        d = dequant(e, data)
        err = float(np.abs(d - t["data"]).max()) if d.size else 0.0
        if e["fmt"] == "f32":
            assert err == 0.0
        elif e["fmt"] in ("i8", "t4"):
            L = e.get("L", 7)
            assert err <= float(np.abs(t["data"]).max()) / L * 0.5 * 1.001 + 1e-7, (e["name"], err)   # fp16 scale rounding
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(data)
    with open(out_dir / "dict.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(chars) + "\n")
    counts = defaultdict(int)
    for op in ex.ops:
        counts[op["op"]] += 1
    manifest = {
        "format": FORMAT,
        "model": name,
        "source": f"PP-OCRv6_{name}_rec (PaddlePaddle inference.onnx, Apache-2.0)",
        "scheme": scheme,
        "scheme_parts": spec,
        "input": {"name": model.graph.input[0].name, "c": 3, "h": int(image_shape[1]), "order": "BGR",
                  "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                  "min_width": int(image_shape[2]), "pad": 0.0},
        "dict": {"file": "dict.txt", "entries": len(chars), "blank": 0, "space_appended": True},
        "V": V,
        "buffers": [{"id": i, "c": c, "kind": kd, "onnx": nm}
                    for i, (c, kd, nm) in enumerate(zip(ex.bufc, ex.bufkind, ex.bufnames))],
        "output": {"buf": out_buf, "tensor": ex.graph_out_name},
        "weights": {"file": "weights.bin", "bytes": len(data), "by_format": wstats},
        "tensors": entries,
        "fusion": dict(ex.stats, onnx_nodes=n_nodes, identity_dropped=ex.n_identity, shape_only_skipped=len(ex.skipped)),
        "ops": ex.ops,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"{name} [{scheme}]: {n_nodes} onnx nodes -> {len(ex.ops)} ops {dict(counts)}; buffers {len(ex.bufc)}; "
          f"V {V}; weights {len(data) / 1e6:.2f} MB {json.dumps(wstats)} -> {out_dir}")
    return manifest


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", default=list(NAMES))
    ap.add_argument("--scheme", default=DEFAULT_SCHEME)
    ap.add_argument("--src", type=Path, default=SRC, help="directory holding <name>_rec/inference.{onnx,yml}")
    ap.add_argument("--out", type=Path, default=None, help="output directory (one model only)")
    a = ap.parse_args(argv)
    for name in a.names:
        if a.out and len(a.names) == 1:
            out = a.out
        else:
            out = OUT_ROOT / (name if a.scheme == DEFAULT_SCHEME else f"{name}-{slug(a.scheme)}")
        export(name, a.src / f"{name}_rec", out, a.scheme)


if __name__ == "__main__":
    main(sys.argv[1:])
