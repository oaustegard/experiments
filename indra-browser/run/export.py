"""Indra's ONNX models as programs and packed weights for the browser engine.

Each model becomes a manifest entry (its op program, its tensors) and one
weights file. A conv net (det, orient, script, mathcls) becomes a straight
list of ops over named NHWC tensors (conv with its activation and residual
fused, up2, s2d, d2s, concat, pool, gmean, gemm); the shape plumbing ONNX
needs (Shape, Gather, Reshape chains) is folded into those ops. The
recognizers and the order model keep their tensors under their ONNX names
(node names for anonymous MatMul weights); their forward passes are written
out in the engine.

Weights: a tensor that quant.py quantizes (Conv / MatMul / Gemm weight of at
least 1,024 values) is stored as b-bit codes in a little-endian bit stream (code + 2^(b-1) - 1
for symmetric schemes; bit-packed codes gzip 8-15% smaller than one byte per code), with fp16 scales per row of its [O, K] view (and a byte
zero point per row for asymmetric ones); everything else is fp16. The engine
turns all of it back into f32 once, at load.

    python3 run/export.py --src /tmp/indra-models --out web/models/s4c --scheme s4c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import quantize, to_ok, weight_axes  # noqa: E402

CONVNETS = ["det", "orient", "script", "mathcls"]
SEQ = ["rec", "rec_world", "rec_cjk", "rec_rtl", "order"]


def _attrs(n):
    out = {}
    for a in n.attribute:
        v = onnx.helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def convnet_program(m) -> tuple[list, str, dict]:
    """Ops over tensor names, the input name, and {weight name: ONNX initializer}."""
    init = {t.name: t for t in m.graph.initializer}
    consumers: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    prod = {o: n for n in m.graph.node for o in n.output}
    alias: dict[str, str] = {}
    ops: list[dict] = []
    weights: dict = {}

    def T(name):
        while name in alias:
            name = alias[name]
        return name

    def last_op_out(name):
        for op in reversed(ops):
            if op.get("out") == name:
                return op
        return None

    for n in m.graph.node:
        t, a = n.op_type, _attrs(n)
        if t == "Conv":
            w, b = n.input[1], (n.input[2] if len(n.input) > 2 else None)
            weights[w] = init[w]
            if b:
                weights[b] = init[b]
            k = a.get("kernel_shape")
            ops.append({"op": "conv", "in": T(n.input[0]), "out": n.output[0], "w": w, "b": b,
                        "k": list(k), "s": list(a.get("strides", [1, 1])), "p": list(a.get("pads", [0, 0, 0, 0])),
                        "g": int(a.get("group", 1)), "act": None, "res": None})
        elif t in ("HardSwish", "Relu", "Sigmoid"):
            src = T(n.input[0])
            op = last_op_out(src)
            if op and op["op"] in ("conv", "gemm") and op["act"] is None and len(consumers.get(src, [])) == 1:
                op["act"] = t.lower()
                alias[n.output[0]] = src
            else:
                ops.append({"op": "act", "act": t.lower(), "in": src, "out": n.output[0]})
        elif t == "Add" and all(i not in init for i in n.input):
            x, y = T(n.input[0]), T(n.input[1])
            # Fuse a residual into the conv that produced one operand (no activation after
            # the add, the conv output used only here).
            fused = False
            for c, r in ((y, x), (x, y)):
                op = last_op_out(c)
                if op and op["op"] == "conv" and op["res"] is None and len(consumers.get(c, [])) == 1 \
                        and prod.get(n.input[0 if c == x else 1]) is not None:
                    op["res"] = r
                    # the residual adds after the activation when one is already fused
                    op["res_at"] = "post" if op["act"] else "pre"
                    alias[n.output[0]] = c
                    # run the conv after the residual's producer (its output is used only here)
                    ops.remove(op)
                    ops.append(op)
                    fused = True
                    break
            if not fused:
                ops.append({"op": "add", "a": x, "b": y, "out": n.output[0]})
        elif t == "Resize":
            assert a.get("mode") == "nearest" and a.get("coordinate_transformation_mode") == "asymmetric"
            ops.append({"op": "up2", "in": T(n.input[0]), "out": n.output[0]})
        elif t == "Reshape" and n.input[0] in prod and prod[n.input[0]].op_type == "Unsqueeze":
            # det's space-to-depth: Unsqueeze -> Reshape -> Reshape -> Transpose -> Reshape -> Squeeze
            src = T(prod[n.input[0]].input[0])
            chain = [n]
            while len(chain) < 5:
                nxt = consumers[chain[-1].output[0]]
                assert len(nxt) == 1, "unexpected s2d chain"
                chain.append(nxt[0])
            assert [c.op_type for c in chain] == ["Reshape", "Reshape", "Transpose", "Reshape", "Squeeze"]
            ops.append({"op": "s2d", "in": src, "out": chain[-1].output[0]})
            for c in chain[:-1]:
                alias[c.output[0]] = chain[-1].output[0]
        elif t == "Concat" and a.get("axis") == 1:
            ops.append({"op": "concat", "ins": [T(i) for i in n.input], "out": n.output[0]})
        elif t == "DepthToSpace":
            ops.append({"op": "d2s", "in": T(n.input[0]), "out": n.output[0],
                        "bs": int(a["blocksize"]), "mode": a.get("mode", "DCR")})
        elif t in ("AveragePool", "MaxPool"):
            assert a["kernel_shape"] == [2, 2] and a["strides"] == [2, 2] and not any(a.get("pads", [0]))
            ops.append({"op": "pool", "kind": "avg" if t == "AveragePool" else "max",
                        "in": T(n.input[0]), "out": n.output[0]})
        elif t == "ReduceMean":
            axes = a.get("axes")
            ops.append({"op": "gmean" if axes == [2, 3] else "hmean", "in": T(n.input[0]), "out": n.output[0]})
        elif t == "Gemm":
            weights[n.input[1]] = init[n.input[1]]
            weights[n.input[2]] = init[n.input[2]]
            assert a.get("transB", 0) == 1
            src = T(n.input[0])
            op = {"op": "gemm", "in": src, "out": n.output[0], "w": n.input[1], "b": n.input[2], "act": None}
            if not any(o.get("out") == src for o in ops) and ops and ops[-1]["op"] == "hmean":
                # script / mathcls: the mean over the valid columns (width // 32, at least 1)
                # that the widths input selects; the engine applies it
                op["in"], op["wdiv"] = ops[-1]["out"], 32
            ops.append(op)
    first = m.graph.input[0].name
    # drop ops whose inputs nothing produced (det's output tail: slice, softplus, concat)
    have = {first}
    keep = []
    for op in ops:
        ins = op.get("ins") or [op.get("in"), op.get("a"), op.get("b") if op["op"] == "add" else None, op.get("res")]
        if all(i is None or i in have for i in ins):
            keep.append(op)
            have.add(op["out"])
    return keep, first, weights


def halves(a, planes: bool) -> bytes:
    """fp16 bytes; with planes, every value's low byte first, then every high byte. The high bytes
    (sign and exponent) repeat a few values, so they gzip much better grouped (5-8% of a file)."""
    b = np.ascontiguousarray(a, np.float16).view(np.uint8).reshape(-1, 2)
    return (b[:, 0].tobytes() + b[:, 1].tobytes()) if planes else b.tobytes()


def tensor_record(name, arr, how, scheme, buf: bytearray, planes: bool = True) -> dict:
    """Append one tensor to buf; its manifest record."""
    def put(b: bytes) -> int:
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(b)
        return off

    rec = {"shape": list(arr.shape)}
    if planes:
        rec["planes"] = 1
    if how is not None and arr.size >= 1024 and arr.ndim >= 2 and scheme not in ("f16", "f32"):
        w2 = to_ok(arr, how)
        r = quantize(w2, scheme)
        q = r["codes"]
        sym = scheme[0] in "is"
        bits = 8 if scheme == "i8c" else int(scheme[1])
        bias = (2 ** (bits - 1) - 1) if sym else 0
        O, K = w2.shape
        G = 0 if scheme == "i8c" or scheme.endswith("c") else int(scheme.split("g")[1])
        if G and G < K:
            ng = -(-K // G)
            codes = q.reshape(O, ng * G)[:, :K]
        else:
            codes, G = q.reshape(O, K), 0
        u = (codes + bias).astype(np.uint32).ravel()
        packed = np.packbits(((u[:, None] >> np.arange(bits)) & 1).astype(np.uint8).ravel(), bitorder="little")
        rec |= {"enc": "q", "view": "conv" if how[0] == "conv" or (how[0] == "gemm" and how[1]) else "matmul",
                "O": O, "K": K, "bits": bits, "sym": sym, "G": G,
                "codes": put(packed.tobytes()),
                "scales": put(halves(r["scales"], planes))}
        if not sym:
            rec["zp"] = put(np.asarray(r.get("zp", np.zeros(len(r["scales"]))), np.uint8).tobytes())
    elif scheme == "f32":
        rec |= {"enc": "f32", "data": put(arr.astype(np.float32).tobytes())}
    else:
        rec |= {"enc": "f16", "data": put(halves(arr, planes))}
    return rec


def export_model(path: Path, scheme: str, out_dir: Path, planes: bool = True) -> dict:
    m = onnx.load(str(path))
    name = path.stem
    axes = weight_axes(m)
    init = {t.name: t for t in m.graph.initializer}
    entry: dict = {"file": f"{name}.bin"}
    if name in CONVNETS:
        ops, first, weights = convnet_program(m)
        entry |= {"kind": "convnet", "input": first, "ops": ops}
        names = list(weights)
        if name == "det":
            entry["post"] = {"softplus_scale": 8.0}
    else:
        entry["kind"] = "rec" if name.startswith("rec") else name
        names = [t.name for t in m.graph.initializer]
        # Anonymous MatMul weights (order) and BN-folded convs (rec dw) get their node's name.
        rename = {}
        for n in m.graph.node:
            if n.op_type in ("MatMul", "Conv") and len(n.input) > 1 and n.input[1] in init \
                    and n.input[1].startswith("onnx::"):
                rename[n.input[1]] = n.name.strip("/").replace("/", ".") + ".weight"
                if n.op_type == "Conv" and len(n.input) > 2 and n.input[2].startswith("onnx::"):
                    rename[n.input[2]] = n.name.strip("/").replace("/", ".") + ".bias"
            if n.op_type == "Add" and any(i in init and i.startswith("onnx::") is False for i in n.input):
                pass
        entry["rename"] = rename
    buf = bytearray()
    tensors = {}
    for tn in names:
        arr = numpy_helper.to_array(init[tn])
        if arr.dtype != np.float32:
            continue
        tensors[entry.get("rename", {}).get(tn, tn)] = tensor_record(tn, arr, axes.get(tn), scheme, buf, planes)
    entry["tensors"] = tensors
    entry.pop("rename", None)
    (out_dir / entry["file"]).write_bytes(bytes(buf))
    entry["bytes"] = len(buf)
    return entry


MATH = ["math_enc", "math_dec_init", "math_dec_step"]


def export_math(src: Path, scheme: str, out_dir: Path, planes: bool = True) -> dict:
    """The math reader (math.py in indra-ocr) as one weights file, fetched only when a page
    has math: the encoder, and the decoder's tensors from its init and step graphs (they
    share the cross-attention projections; each tensor is stored once). Anonymous tensors
    get names: a conv's from its node ("enc.blocks.0.dw.Conv.weight"), a MatMul weight from
    the bias added to its output ("...linear1.bias" -> "...linear1.weight"). The decoder's
    "c." prefix is dropped. Its forward passes are written out in web/math.js."""
    buf = bytearray()
    tensors: dict = {}
    for name in MATH:
        m = onnx.load(str(src / f"{name}.onnx"))
        axes = weight_axes(m)
        init = {t.name: t for t in m.graph.initializer}
        cons: dict = {}
        for n in m.graph.node:
            for i in n.input:
                cons.setdefault(i, []).append(n)
        rename = {}
        for n in m.graph.node:
            if n.op_type == "Conv" and n.input[1].startswith("onnx::"):
                base = n.name.strip("/").replace("/", ".")
                rename[n.input[1]] = base + ".weight"
                if len(n.input) > 2:
                    rename[n.input[2]] = base + ".bias"
            elif n.op_type == "MatMul" and n.input[1] in init and n.input[1].startswith("onnx::"):
                bias = [i for c in cons.get(n.output[0], []) if c.op_type == "Add" for i in c.input if i in init]
                assert len(bias) == 1 and bias[0].endswith("bias"), (name, n.name)
                rename[n.input[1]] = bias[0][: -len("bias")] + "weight"
        for tn, t in init.items():
            arr = numpy_helper.to_array(t)
            if arr.dtype != np.float32:
                continue
            key = rename.get(tn, tn)
            key = key[2:] if key.startswith("c.") else key
            if key in tensors:
                continue
            tensors[key] = tensor_record(tn, arr, axes.get(tn), scheme, buf, planes)
    (out_dir / "math.bin").write_bytes(bytes(buf))
    meta = json.loads((src / "math_meta.json").read_text())
    return {"file": "math.bin", "kind": "math", "scheme": scheme, "tensors": tensors, "bytes": len(buf),
            "config": meta["config"], "tokens": meta["tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scheme", default="s4c")
    ap.add_argument("--per", default="", help="model=scheme,... overrides")
    ap.add_argument("--models", default="det,orient,script,rec,rec_world,rec_cjk,rec_rtl,order")
    ap.add_argument("--lm", default="", help="n-gram tables: base=ORDER:VBITS,world=...,cjk=... "
                    "(ORDER 0 keeps every order, VBITS 32 keeps float32 values); empty: none")
    ap.add_argument("--no-planes", action="store_true", help="fp16 bytes interleaved, as packs before 2026-11-18")
    ap.add_argument("--math", default="", help="the math reader's scheme (f16, i8c, ...): mathcls.bin (fp16) "
                    "and math.bin, under the manifest's \"math\" key, outside the core models; empty: none")
    a = ap.parse_args()
    per = dict(kv.split("=") for kv in a.per.split(",") if kv)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.src / "meta.json").read_text())
    man = {"format": "indra-pack/1", "scheme": a.scheme, "per": per,
           "source": "datalab-to/indra-ocr@v1.1.0", "models": {},
           "det_long_side": meta["det_long_side"], "rec_height": meta["rec_height"],
           "rec_window": meta.get("rec_window", 1), "script_groups": meta.get("script_groups"),
           "lm_params": meta.get("lm_params", {})}
    for name in a.models.split(","):
        e = export_model(a.src / f"{name}.onnx", per.get(name, a.scheme), a.out, not a.no_planes)
        man["models"][name] = e
        print(f"{name}: {e['bytes']/1e3:.0f} KB, {len(e['tensors'])} tensors" +
              (f", {len(e['ops'])} ops" if "ops" in e else ""))
    if a.lm:
        from lmpack import write_trie as write_lm

        man["lm"] = {}
        for spec in a.lm.split(","):
            name, ov = spec.split("=")
            order, vbits = (int(v) for v in ov.split(":"))
            src = a.src / ("lm.npz" if name == "base" else f"lm_{name}.npz")
            dst = "lm.bin" if name == "base" else f"lm_{name}.bin"
            write_lm(str(src), str(a.out / dst), order or None, vbits)
            man["lm"][name] = dst
            print(f"lm {name}: order {order or 'all'}, {vbits}-bit values, {(a.out / dst).stat().st_size / 1e3:.0f} KB")
    if a.math:
        cls = export_model(a.src / "mathcls.onnx", "f16", a.out, not a.no_planes)
        net = export_math(a.src, a.math, a.out, not a.no_planes)
        man["math"] = {"models": {"mathcls": cls, "math": net}}
        print(f"mathcls: {cls['bytes']/1e3:.0f} KB, {len(cls['ops'])} ops; math ({a.math}): "
              f"{net['bytes']/1e6:.2f} MB, {len(net['tensors'])} tensors")
    for k in [k for k in meta if k.startswith("vocab")]:
        (a.out / f"{k}.json").write_text(json.dumps(json.loads(meta[k])["chars"], ensure_ascii=False))
    (a.out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
