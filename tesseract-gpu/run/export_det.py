#!/usr/bin/env python3
"""Export the PP-OCRv6 text-line detector (ONNX) to a straight op program plus packed weights.

    python3 -I run/export_det.py            # tiny and small -> web/det/models/<name>/

The graph is walked once in topological order. Single-consumer chains fold into one op:

  Conv [+ BatchNormalization] [+ bias] + Relu | HardSigmoid | Sigmoid | exact GELU  -> "conv"
  ReduceMean(axes=[2,3]) or GlobalAveragePool -> Conv1x1 -> Relu -> Conv1x1 -> HardSigmoid
        -> Mul(x, gate) [-> Add(x, .)]                                              -> "se"
  ConvTranspose [+ bias] [+ Relu | Sigmoid]                                          -> "convt"
  MaxPool, Concat(axis 1), Resize(nearest, integer scale), Add(tensor, tensor)       -> as is

GELU is the exact form Div(x, sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(., 0.5).
A LayerNorm (ReduceMean over the channel axis with Sub/Pow/Sqrt) is not in these graphs; any
ReduceMean that is not a squeeze-excite block is an error, so none can slip through.

Weights: one file of u32 words. Each conv/ConvTranspose weight tensor is stored as two f16
planes (hi, then lo = the rounding residual), low half of a word first; the dequantised weight is
hi + lo. Biases are f32 bit patterns. Offsets in the manifest are word offsets; "wl" is the
element distance from the hi plane to the lo plane.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"tiny": ROOT / "models/det/tiny", "small": ROOT / "models/det/small"}
OUT_ROOT = ROOT / "web/det/models"
FORMAT = "tesseract-gpu/det-program v2"
SQRT2 = 1.4142135
TOL = 1e-5
ACT_CODE = {"none": 0, "relu": 1, "hsig": 2, "gelu": 3, "sigmoid": 4}


def attr(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def txt(v):
    return v.decode() if isinstance(v, bytes) else v


class Packer:
    def __init__(self):
        self.chunks, self.words = [], 0

    def f16_pair(self, arr):
        """Weights as two f16 planes: hi = f16(w), lo = f16(w - hi). Returns (word offset, plane stride).

        One f16 plane is 2^-11 relative per weight, which compounds to about 2e-2 on the output
        probability map (measured); the residual plane brings the dequantised weight to f32
        precision. Plane stride is the element distance from hi to lo (even, so word-aligned).
        """
        a = np.asarray(arr, dtype=np.float32).ravel()
        hi = a.astype("<f2")
        lo = (a - hi.astype(np.float32)).astype("<f2")
        n = a.size + (a.size % 2)
        hi = np.concatenate([hi, np.zeros(n - a.size, "<f2")])
        lo = np.concatenate([lo, np.zeros(n - a.size, "<f2")])
        off = self.words
        self.chunks.append(hi.tobytes() + lo.tobytes())
        self.words += n
        return off, n

    def f32(self, arr):
        a = np.asarray(arr, dtype="<f4").ravel()
        off = self.words
        self.chunks.append(a.tobytes())
        self.words += a.size
        return off

    def data(self):
        return b"".join(self.chunks)


class Exporter:
    def __init__(self, model):
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
        self.pack = Packer()
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

    def conv_entry(self, W, B, act, ha, hb, geom):
        w, wl = self.pack.f16_pair(W)
        return dict(geom, act=act, ha=ha, hb=hb, w=w, wl=wl,
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
        e = self.conv_entry(W, B, act, ha, hb, geom)
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


def normalize_from_yaml(path):
    cfg = yaml.safe_load(open(path))
    norm = next(op["NormalizeImage"] for op in cfg["PreProcess"]["transform_ops"] if "NormalizeImage" in op)
    assert norm.get("order") == "hwc", norm
    return [float(v) for v in norm["mean"]], [float(v) for v in norm["std"]], norm["scale"]


def export(name, src_dir, out_dir):
    model = onnx.load(str(src_dir / "inference.onnx"))
    ex = Exporter(model)
    out = ex.run()
    mean, std, scale = normalize_from_yaml(src_dir / "inference.yml")
    assert str(scale).replace(" ", "") in ("1./255.", "1/255"), scale
    data = ex.pack.data()
    assert len(data) % 4 == 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(data)
    manifest = {
        "format": FORMAT,
        "model": name,
        "source": f"PP-OCRv6_{name}_det (PaddleOCR inference.onnx, Apache-2.0)",
        "input": {"name": "x", "c": 3, "order": "BGR", "mean": mean, "std": std,
                  "scale": 1 / 255, "multiple": 32, "default_long_side": 1280},
        "buffers": [{"id": i, "c": c, "onnx": nm} for i, (c, nm) in enumerate(zip(ex.bufc, ex.bufnames))],
        "output": {"buf": ex.buf[out], "c": 1, "tensor": out},
        "weights": {"file": "weights.bin", "bytes": len(data)},
        "fusion": dict(ex.stats),
        "ops": ex.ops,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    counts = defaultdict(int)
    for op in ex.ops:
        counts[op["op"]] += 1
    acts = defaultdict(int)
    for op in ex.ops:
        if op["op"] in ("conv", "convt"):
            acts[op["act"]] += 1
    print(f"{name}: {len(model.graph.node)} onnx nodes -> {len(ex.ops)} ops {dict(counts)}; "
          f"conv/convt activations {dict(acts)}; se blocks {ex.stats['se']}; "
          f"buffers {len(ex.bufc)}; weights {len(data)/1e6:.2f} MB -> {out_dir}")


def main(argv):
    names = argv or list(MODELS)
    for name in names:
        export(name, MODELS[name], OUT_ROOT / name)


if __name__ == "__main__":
    main(sys.argv[1:])
