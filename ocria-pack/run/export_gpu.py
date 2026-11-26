#!/usr/bin/env python3
"""Export OCRia-nano-1's OCR nets (ONNX) to a straight op program plus packed weights for the WGSL engine
in web/gpu/ (engine.js runs the program; the parity test in web/gpu/test/ checks it against onnxruntime).

    python3 -I run/export_gpu.py                  # every model at L=127 -> web/gpu/models/<model>/
    python3 -I run/export_gpu.py --levels 63 det  # one model, another level
    python3 -I run/export_gpu.py --sizes          # bytes and gzip bytes at L=127 and L=63 -> results/gpu_sizes.json

The graph is walked once in node order. Each node becomes an op, or folds into the op that produced its
input when that op's output has no other reader (the fused op moves to the folded node's place, which is
safe because nothing else reads the value in between). A conv-like op ("conv", "pw", "gemm", "add") ends in
an epilogue applied in this order:

    v = acc + bias;  v = act1(v);  v = fq1(v);  v = v + res;  v = act2(v);  v = fq2(v)

act is relu or sigmoid; fq(v) = (clamp(round_half_even(v / s) + z, 0, 255) - z) * s, the value an
activation QuantizeLinear -> DequantizeLinear pair (uint8, per tensor) gives. Folding rules:

  Relu | Sigmoid                       -> act1, or act2 once a residual is in
  QuantizeLinear -> DequantizeLinear   -> fq1, or fq2 once a residual is in; an fq repeating the fq its
                                          input already went through (after Squeeze) is dropped
  Add(a, b)                            -> res of the op producing a (or b)
  Mul(x, c[1,3,1,1]) -> ReduceSum(1)   -> "gray"
  Squeeze(axis 2)                      -> no op (NCHW with H = 1 is NCT)
  ReduceMean(2), ReduceMax(2) -> Concat -> "meanmax"  (router)
  Conv 1x1 -> {ArgMax, ReduceMax, ReduceLogSumExp}(1) -> Sub -> Exp  -> "head" (ids, conf), CTC heads

Ops: conv (direct, any groups; 1-D convs are 2-D with H = 1), pw (1x1, group 1, stride 1: a GEMM), gemm
(Gemm transB = 1 on [N, C]), add, resize (nearest, integer scale, asymmetric floor), concat (channels),
gray, fq, meanmax, head. Any other op is an error.

Weights: one file, every blob 4-byte aligned, little-endian, all int8 codes first and then all float32
(the two kinds compress better apart). A tensor is

  f32  float32 values at byte offset "off"
  i8   int8 codes at "off", row-major with rows = dim 0 (a conv's or Gemm's output channel), then one
       float32 scale per row at "soff"; value = code * scale

Float tensors of 1024 or more elements are i8 with codes in -L..L, scale = max|row| / L, code =
rint(w / scale) (run/pack.py's scheme); biases and every smaller tensor are f32. rec_latin is already a
QDQ graph: its weights are taken as they are (int8 codes in -127..127 and float32 per-channel scales, zero
points all 0), its int32 biases are dequantized as onnxruntime does ((b - z) * s in float32), and L does
not apply to it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ["det", "router", "rec_latin", "rec_world", "rec_rtl", "rec_cjk"]
FORMAT = "ocria-gpu/program v1"
MIN_ELEMS = 1024


def quantize_rows(w: np.ndarray, levels: int):
    """run/pack.py's scheme: int8 codes per row (dim 0), float32 scale per row."""
    rows = w.shape[0] if w.ndim > 1 else 1
    w2 = w.reshape(rows, -1).astype(np.float32)
    amax = np.abs(w2).max(axis=1)
    scale = np.where(amax > 0, amax / levels, 1.0).astype(np.float32)
    q = np.clip(np.rint(w2 / scale[:, None]), -levels, levels).astype(np.int8)
    return q.reshape(w.shape), scale


class Weights:
    def __init__(self, levels: int):
        self.levels = levels
        self.items = {}   # name -> ("i8", codes, scales) | ("f32", array)

    def add_float(self, name: str, w: np.ndarray, bias: bool = False) -> str:
        w = np.asarray(w, np.float32)
        if bias or w.size < MIN_ELEMS:
            self.items[name] = ("f32", w)
        else:
            q, s = quantize_rows(w, self.levels)
            self.items[name] = ("i8", q, s)
        return name

    def add_i8(self, name: str, codes: np.ndarray, scales: np.ndarray) -> str:
        self.items[name] = ("i8", codes.astype(np.int8), scales.astype(np.float32))
        return name

    def pack(self):
        """(bytes, manifest entries). int8 region first, then float32 region."""
        q, f = bytearray(), bytearray()
        man = {}
        for name, it in self.items.items():
            if it[0] == "i8":
                man[name] = {"kind": "i8", "shape": list(it[1].shape), "off": len(q)}
                q += it[1].tobytes()
                q += b"\0" * (-len(q) % 4)
        for name, it in self.items.items():
            if it[0] == "i8":
                man[name]["soff"] = len(f)
                f += it[2].astype("<f4").tobytes()
            else:
                man[name] = {"kind": "f32", "shape": list(it[1].shape), "off": len(f)}
                f += it[1].astype("<f4").tobytes()
        base = len(q)
        for e in man.values():
            if e["kind"] == "f32":
                e["off"] += base
            else:
                e["soff"] += base
        return bytes(q) + bytes(f), man

    def dequantized(self) -> dict[str, np.ndarray]:
        """Float32 value of every tensor as the engine sees it (code * scale in float32)."""
        out = {}
        for name, it in self.items.items():
            if it[0] == "f32":
                out[name] = it[1]
            else:
                c, s = it[1], it[2]
                out[name] = (c.reshape(c.shape[0], -1).astype(np.float32) * s[:, None]).reshape(c.shape)
        return out


def attrs(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def export(path: Path, levels: int, name: str):
    m = onnx.load(str(path))
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consts = {}
    W = Weights(levels)
    deq = {}          # name -> ("i8", codes, scales) | ("f32", values) from weight DequantizeLinear nodes
    pending_q = {}    # QuantizeLinear output -> (x, s, z)

    nreaders = {}
    for n in g.node:
        for i in n.input:
            if i:
                nreaders[i] = nreaders.get(i, 0) + 1
    graph_outs = {o.name for o in g.output}
    for o in graph_outs:
        nreaders[o] = nreaders.get(o, 0) + 1

    tensors = [{"c": int(g.input[0].type.tensor_type.shape.dim[1].dim_value)}]
    tid = {g.input[0].name: 0}
    last_fq = {}      # tensor id -> (s, z) of the fq it last went through
    ops = []
    prod = {}         # tensor name -> op, for ops whose output has exactly one reader
    handled = set()

    def new_tensor(nm, c, **kw):
        tensors.append({"c": int(c), **kw})
        tid[nm] = len(tensors) - 1
        return tid[nm]

    def emit(op, out_name):
        ops.append(op)
        if nreaders.get(out_name, 0) == 1 and out_name not in graph_outs:
            prod[out_name] = op

    def take(in_name):
        """The op producing in_name if it can absorb its reader, else None."""
        op = prod.get(in_name)
        if op is None or op["op"] not in ("conv", "pw", "gemm", "add"):
            return None
        return op

    def move(op, old_name, new_name):
        ops.remove(op)
        ops.append(op)
        prod.pop(old_name, None)
        tid[new_name] = op["out"]
        if nreaders.get(new_name, 0) == 1 and new_name not in graph_outs:
            prod[new_name] = op

    def stage(op):
        if op.get("fq2"):
            return 5
        if op.get("act2"):
            return 4
        if op.get("res") is not None or op["op"] == "add":
            return 3
        if op.get("fq1"):
            return 2
        if op.get("act1"):
            return 1
        return 0

    def weight(nm, bias=False):
        if nm in deq:
            it = deq[nm]
            if it[0] == "i8":
                return W.add_i8(nm, it[1], it[2])
            W.items[nm] = ("f32", it[1].astype(np.float32))
            return nm
        return W.add_float(nm, init[nm], bias=bias)

    def fuse_unary(n, kind, val):
        x, y = n.input[0], n.output[0]
        op = take(x)
        if op is not None:
            st = stage(op)
            if kind == "act" and st == 0:
                op["act1"] = val
            elif kind == "act" and st == 3:
                op["act2"] = val
            elif kind == "fq" and st <= 1:
                op["fq1"] = val
            elif kind == "fq" and st in (3, 4):
                op["fq2"] = val
            else:
                op = None
            if op is not None:
                if kind == "fq":
                    last_fq[op["out"]] = tuple(val)
                move(op, x, y)
                return
        if kind == "fq":
            if last_fq.get(tid[x]) == tuple(val):     # repeats the fq already applied: no op
                tid[y] = tid[x]
                return
            out = new_tensor(y, tensors[tid[x]]["c"])
            last_fq[out] = tuple(val)
            emit({"op": "fq", "in": tid[x], "out": out, "fq": list(val)}, y)
            return
        raise ValueError(f"{name}: cannot fold {n.op_type} {n.name}")

    nodes = list(g.node)
    # CTC head: ids = ArgMax(L, 1); conf = Exp(Sub(ReduceMax(L, 1), ReduceLogSumExp(L, 1)))
    by_out = {o: n for n in nodes for o in n.output}
    head = None
    if "ids" in graph_outs:
        am = by_out["ids"]
        ex = by_out["conf"]
        sub = by_out[ex.input[0]]
        rmax, lse = by_out[sub.input[0]], by_out[sub.input[1]]
        assert am.op_type == "ArgMax" and attrs(am).get("axis") == 1 and attrs(am).get("select_last_index", 0) == 0
        assert ex.op_type == "Exp" and sub.op_type == "Sub" and rmax.op_type == "ReduceMax"
        assert lse.op_type == "ReduceLogSumExp" and am.input[0] == rmax.input[0] == lse.input[0]
        assert list(attrs(rmax)["axes"]) == [1] and list(attrs(lse)["axes"]) == [1]
        head = am.input[0]
        handled.update(id(n) for n in (am, ex, sub, rmax, lse))
        nreaders[head] = 1   # the head op is its one reader

    pool_parts = {}   # ReduceMean / ReduceMax output -> (kind, input)
    gray_mul = {}

    for n in nodes:
        if id(n) in handled:
            continue
        t = n.op_type
        a = attrs(n)
        if t == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(a["value"])
        elif t == "DequantizeLinear" and n.input[0] in init:
            q = init[n.input[0]]
            s = init[n.input[1]].astype(np.float32)
            z = init[n.input[2]] if len(n.input) > 2 and n.input[2] else np.zeros_like(s, dtype=q.dtype)
            assert int(np.abs(z).max()) == 0, "weight zero points must be 0"
            assert a.get("axis", 1) == 0 or s.size == 1
            if q.dtype == np.int8:
                assert s.size == q.shape[0]
                deq[n.output[0]] = ("i8", q, s)
            else:  # int32 bias
                sh = (-1,) + (1,) * (q.ndim - 1)
                deq[n.output[0]] = ("f32", (q - z).astype(np.float32) * s.reshape(sh) if s.size > 1 else (q - z).astype(np.float32) * s)
        elif t == "QuantizeLinear":
            s = float(init[n.input[1]])
            z = init[n.input[2]]
            assert z.dtype == np.uint8 and z.size == 1
            pending_q[n.output[0]] = (n.input[0], s, int(z))
        elif t == "DequantizeLinear":
            x, s, z = pending_q.pop(n.input[0])
            assert abs(float(init[n.input[1]]) - s) == 0 and int(init[n.input[2]]) == z
            fake = helper.make_node("FQ", [x], [n.output[0]])
            fuse_unary(fake, "fq", [np.float32(s).item(), z])
        elif t == "Conv":
            x, wn = n.input[0], n.input[1]
            bn = n.input[2] if len(n.input) > 2 and n.input[2] else None
            wv = deq[wn][1] if wn in deq else init[wn]
            k = list(a["kernel_shape"])
            s = list(a.get("strides", [1] * len(k)))
            p = list(a.get("pads", [0] * 2 * len(k)))
            d = list(a.get("dilations", [1] * len(k)))
            assert all(v == 1 for v in d), "dilation"
            grp = int(a.get("group", 1))
            if len(k) == 1:
                k, s, p = [1, k[0]], [1, s[0]], [0, p[0], 0, p[1]]
            cout, icg = int(wv.shape[0]), int(wv.shape[1])
            cin = icg * grp
            assert cin == tensors[tid[x]]["c"], f"{name}: conv input channels"
            if bn is None:
                bname = n.output[0] + ".zero_bias"
                W.items[bname] = ("f32", np.zeros(cout, np.float32))
            else:
                bname = weight(bn, bias=True)
            wname = weight(wn)
            out = new_tensor(n.output[0], cout)
            if k == [1, 1] and s == [1, 1] and grp == 1 and not any(p):
                op = {"op": "pw", "in": tid[x], "out": out, "cin": cin, "cout": cout, "w": wname, "b": bname}
            else:
                op = {"op": "conv", "in": tid[x], "out": out, "cin": cin, "cout": cout, "g": grp,
                      "k": k, "s": s, "p": p, "w": wname, "b": bname}
            if n.output[0] == head:
                op["head_of"] = True
            emit(op, n.output[0])
        elif t in ("Relu", "Sigmoid"):
            fuse_unary(n, "act", t.lower())
        elif t == "Add":
            a0, a1 = n.input
            y = n.output[0]
            done = False
            for mine, other in ((a0, a1), (a1, a0)):
                op = take(mine)
                if op is not None and op["op"] != "add" and stage(op) <= 2 and tid[other] != op["out"]:
                    op["res"] = tid[other]
                    move(op, mine, y)
                    done = True
                    break
            if not done:
                assert tensors[tid[a0]]["c"] == tensors[tid[a1]]["c"]
                out = new_tensor(y, tensors[tid[a0]]["c"])
                emit({"op": "add", "ins": [tid[a0], tid[a1]], "out": out}, y)
        elif t == "Resize":
            sc = consts.get(n.input[2]) if len(n.input) > 2 else None
            assert sc is not None and list(sc[:2]) == [1, 1] and sc[2] == sc[3] and float(sc[2]).is_integer()
            assert a.get("mode", b"nearest") in (b"nearest", "nearest")
            assert a.get("coordinate_transformation_mode") in (b"asymmetric", "asymmetric")
            assert a.get("nearest_mode") in (b"floor", "floor")
            x = n.input[0]
            out = new_tensor(n.output[0], tensors[tid[x]]["c"])
            emit({"op": "resize", "in": tid[x], "out": out, "scale": int(sc[2])}, n.output[0])
        elif t == "Mul":
            c = consts.get(n.input[1])
            assert c is not None and c.shape == (1, 3, 1, 1), f"{name}: Mul other than the gray weighting"
            gray_mul[n.output[0]] = (n.input[0], [float(v) for v in c.ravel()])
        elif t == "ReduceSum":
            x, wts = gray_mul.pop(n.input[0])
            axes = consts[n.input[1]].tolist()
            assert axes == [1] and a.get("keepdims", 1) == 1
            out = new_tensor(n.output[0], 1)
            emit({"op": "gray", "in": tid[x], "out": out, "coef": wts}, n.output[0])
        elif t == "Squeeze":
            axes = consts[n.input[1]].tolist()
            assert axes == [2]
            x = n.input[0]
            tid[n.output[0]] = tid[x]
            prod.pop(x, None)
            # a reader of the squeezed tensor sees H = 1; the engine checks it
            tensors[tid[x]]["h1"] = True
        elif t in ("ReduceMean", "ReduceMax"):
            assert list(a["axes"]) == [2] and a.get("keepdims", 1) == 0
            pool_parts[n.output[0]] = (t, n.input[0])
        elif t == "Concat":
            assert a["axis"] == 1
            if all(i in pool_parts for i in n.input):
                (k0, x0), (k1, x1) = pool_parts[n.input[0]], pool_parts[n.input[1]]
                assert (k0, k1) == ("ReduceMean", "ReduceMax") and x0 == x1 and len(n.input) == 2
                out = new_tensor(n.output[0], 2 * tensors[tid[x0]]["c"])
                emit({"op": "meanmax", "in": tid[x0], "out": out}, n.output[0])
            else:
                ins = [tid[i] for i in n.input]
                out = new_tensor(n.output[0], sum(tensors[i]["c"] for i in ins))
                emit({"op": "concat", "ins": ins, "out": out}, n.output[0])
        elif t == "Gemm":
            assert a.get("transB", 0) == 1 and a.get("transA", 0) == 0
            assert a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0
            x = n.input[0]
            wv = init[n.input[1]]
            out = new_tensor(n.output[0], wv.shape[0])
            emit({"op": "gemm", "in": tid[x], "out": out, "cin": int(wv.shape[1]), "cout": int(wv.shape[0]),
                  "w": weight(n.input[1]), "b": weight(n.input[2], bias=True)}, n.output[0])
        else:
            raise ValueError(f"{name}: unsupported op {t} ({n.name})")

    assert not pending_q and not gray_mul, "unpaired QuantizeLinear or Mul"
    outputs = {}
    if head is not None:
        op = next(o for o in ops if o["out"] == tid[head])
        assert op["op"] == "pw" and not op.get("act1") and op.get("res") is None and not op.get("fq2")
        op["op"] = "head"
        op.pop("head_of", None)
        ids = new_tensor("ids", 1, dtype="i32")
        conf = new_tensor("conf", 1)
        op["out"] = [ids, conf]
        outputs = {"ids": {"tensor": ids, "rank": 2}, "conf": {"tensor": conf, "rank": 2}}
    for o in g.output:
        if o.name in outputs:
            continue
        rank = len(o.type.tensor_type.shape.dim)
        outputs[o.name] = {"tensor": tid[o.name], "rank": rank}
    for op in ops:
        op.pop("head_of", None)
        for f in ("act1", "act2", "fq1", "fq2", "res"):
            if op.get(f) is None:
                op.pop(f, None)

    blob, wman = W.pack()
    in_dims = [d.dim_value or d.dim_param for d in g.input[0].type.tensor_type.shape.dim]
    prog = {
        "format": FORMAT,
        "model": name,
        "levels": None if name == "rec_latin" else levels,
        "input": {"name": g.input[0].name, "shape": in_dims},
        "tensors": tensors,
        "ops": ops,
        "outputs": outputs,
        "weights": {"file": "weights.bin", "bytes": len(blob), "tensors": wman},
    }
    return prog, blob, W


def gz(b: bytes) -> int:
    return len(gzip.compress(b, 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=MODELS)
    ap.add_argument("--src", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "web/gpu/models")
    ap.add_argument("--levels", type=int, default=127)
    ap.add_argument("--sizes", action="store_true", help="byte counts at L=127 and L=63 -> results/gpu_sizes.json")
    a = ap.parse_args()
    if a.sizes:
        res = {"note": "program.json + weights.bin per model; gzip -9; upstream = the ONNX file", "models": {}}
        tot = {}
        for nm in a.models:
            src = a.src / f"{nm}.onnx"
            up = src.read_bytes()
            r = {"upstream_bytes": len(up), "upstream_gzip_bytes": gz(up)}
            for L in (127, 63):
                prog, blob, _ = export(src, L, nm)
                pj = json.dumps(prog, separators=(",", ":")).encode()
                r[f"L{L}"] = {"weights_bytes": len(blob), "weights_gzip_bytes": gz(blob),
                              "program_bytes": len(pj), "program_gzip_bytes": gz(pj),
                              "total_gzip_bytes": gz(blob) + gz(pj)}
                tot.setdefault(f"L{L}", 0)
                tot[f"L{L}"] += gz(blob) + gz(pj)
            tot.setdefault("upstream", 0)
            tot["upstream"] += r["upstream_gzip_bytes"]
            res["models"][nm] = r
            print(f"{nm:10s} upstream {len(up):>9,} B (gz {r['upstream_gzip_bytes']:>9,})  "
                  f"L127 {r['L127']['weights_bytes']:>9,} B (gz {r['L127']['total_gzip_bytes']:>9,})  "
                  f"L63 {r['L63']['weights_bytes']:>9,} B (gz {r['L63']['total_gzip_bytes']:>9,})")
        res["total_gzip_bytes"] = tot
        print("total gzip:", tot)
        out = ROOT / "results/gpu_sizes.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + "\n")
        return
    for nm in a.models:
        prog, blob, _ = export(a.src / f"{nm}.onnx", a.levels, nm)
        d = a.out / nm
        d.mkdir(parents=True, exist_ok=True)
        (d / "weights.bin").write_bytes(blob)
        (d / "program.json").write_text(json.dumps(prog, separators=(",", ":")) + "\n")
        for ext in ("charset", "meta"):   # what the page logic needs beside the net (decoding, threshold)
            side = a.src / f"{nm}.{ext}.json"
            if side.exists():
                (d / f"{ext}.json").write_bytes(side.read_bytes())
        kinds = {}
        for op in prog["ops"]:
            kinds[op["op"]] = kinds.get(op["op"], 0) + 1
        print(f"{nm}: {len(prog['ops'])} ops {kinds}; weights {len(blob):,} B (gzip {gz(blob):,})")


if __name__ == "__main__":
    sys.exit(main())
