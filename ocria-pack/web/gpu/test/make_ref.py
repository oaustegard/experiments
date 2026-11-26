#!/usr/bin/env python3
"""Fixtures and onnxruntime references for the WGSL engine's parity check (run_parity.mjs).

    python3 -I web/gpu/test/make_ref.py            # -> web/gpu/test/data/ (gitignored)

Inputs (all real pixels from /home/user/opb/gray600, or $OCRIA_PAGES):
  det     3 page images with a long side of at least 1600 px, area-resized as LiteParse's detect_at does to
          a long side of 1600 (each side rounded to a multiple of 32), gray replicated to RGB, pixel/127.5 - 1.
  lines   320 line crops cut from 10 pages by the detector itself (onnxruntime, LiteParse's threshold 0.55,
          min score 0.6, 4-connected components, the 0.3 shrink undone), prepared as rec.rs does:
          router  Router::prep (height 32, width from the aspect ratio, tiled across 512), batches of 64
          rec     Recognizer::read (height 32, width clamped to [8, 1280]), sorted by width, batches of 32,
                  padded with 1.0 (white) to a multiple of 128
References, onnxruntime 1.x CPU, default session options (ORT_ENABLE_ALL, as LiteParse's native build):
  ref     the ONNX graph with every weight replaced by the value the engine computes from
          web/gpu/models/<model>/weights.bin (the unpacked-equivalent weights). rec_latin's weights are its
          own int8 codes, so its ref is the upstream QDQ graph unchanged.
  orig    the upstream fp32 weights (the cost of packing; fp32 models only)
  noopt   rec_latin only: the QDQ graph with graph optimizations off (ORT_DISABLE_ALL), i.e. plain
          float fake-quant, which is what the WGSL engine computes
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from scipy import ndimage

HERE = Path(__file__).resolve().parent
GPU = HERE.parent
PACK = GPU.parent.parent
OUT = HERE / "data"
PAGES = Path(os.environ.get("OCRIA_PAGES", "/home/user/opb/gray600"))
MODELS = ["det", "router", "rec_latin", "rec_world", "rec_rtl", "rec_cjk"]
LONG = 1600
N_LINES = 320


def engine_weights(m: str) -> dict[str, np.ndarray]:
    prog = json.loads((GPU / "models" / m / "program.json").read_text())
    raw = (GPU / "models" / m / "weights.bin").read_bytes()
    out = {}
    for k, e in prog["weights"]["tensors"].items():
        n = int(np.prod(e["shape"]))
        if e["kind"] == "f32":
            out[k] = np.frombuffer(raw, "<f4", n, e["off"]).reshape(e["shape"])
        else:
            rows = e["shape"][0]
            c = np.frombuffer(raw, np.int8, n, e["off"]).reshape(rows, -1)
            s = np.frombuffer(raw, "<f4", rows, e["soff"])
            out[k] = (c.astype(np.float32) * s[:, None]).reshape(e["shape"])
    return out


def session(m: str, kind: str):
    src = PACK / "models" / f"{m}.onnx"
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    if kind == "noopt":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if kind == "ref" and m != "rec_latin":
        mo = onnx.load(str(src))
        W = engine_weights(m)
        n = 0
        for t in mo.graph.initializer:
            if t.name in W and t.data_type == onnx.TensorProto.FLOAT:
                t.CopyFrom(numpy_helper.from_array(W[t.name].astype(np.float32).reshape(tuple(t.dims)), t.name))
                n += 1
        assert n > 0
        return ort.InferenceSession(mo.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])


def det_input(gray: np.ndarray, long_side: int):
    """detect_at: scale <= 1, sides rounded half-even to multiples of 32, area resize, pixel/127.5 - 1."""
    h, w = gray.shape
    s = min(long_side / max(h, w), 1.0)
    H = max(32, int(np.round(h * s / 32)) * 32)
    W = max(32, int(np.round(w * s / 32)) * 32)
    small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / np.float32(127.5) - np.float32(1.0)), W / w, H / h


def boxes_from_prob(prob: np.ndarray, sx: float, sy: float, img_w: int, img_h: int, thresh=0.55, min_score=0.6):
    """Axis-aligned version of detect_at's box step."""
    ph, pw = prob.shape
    stride = 2.0
    lab, n = ndimage.label(prob > thresh)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        comp = lab[sl] == i + 1
        if comp.sum() < 2 or prob[sl][comp].mean() < min_score:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        kh = (y1 - y0) * stride
        d = 0.3 * kh / 0.4
        bx0, bx1 = (x0 * stride - d) / sx, (x1 * stride + d) / sx
        by0, by1 = (y0 * stride - d) / sy, (y1 * stride + d) / sy
        bx0, bx1 = max(0, int(bx0)), min(img_w, int(np.ceil(bx1)))
        by0, by1 = max(0, int(by0)), min(img_h, int(np.ceil(by1)))
        if bx1 - bx0 >= 8 and by1 - by0 >= 6:
            out.append((by0, (bx0, by0, bx1, by1)))
    out.sort()
    return [b for _, b in out]


def prep_line(crop: np.ndarray, min_w: int, max_w: int) -> np.ndarray:
    h, w = crop.shape
    nw = int(np.clip(np.round(w * 32 / max(h, 1)), min_w, max_w))
    interp = cv2.INTER_AREA if h > 32 else cv2.INTER_LINEAR
    return cv2.resize(crop, (nw, 32), interpolation=interp)


def save(name: str, a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    (OUT / name).write_bytes(a.tobytes())
    return {"file": name, "dtype": str(a.dtype), "shape": list(a.shape)}


def main():
    OUT.mkdir(exist_ok=True)
    for f in OUT.iterdir():
        f.unlink()
    pages = sorted(PAGES.glob("*.png"))
    cv2.setNumThreads(1)
    sess = {}
    for m in MODELS:
        sess[(m, "ref")] = session(m, "ref")
        sess[(m, "orig" if m != "rec_latin" else "noopt")] = session(m, "orig" if m != "rec_latin" else "noopt")
    cases = []

    def run(m, kind, x):
        s = sess[(m, kind)]
        names = [o.name for o in s.get_outputs()]
        return dict(zip(names, s.run(None, {"x": x})))

    # det: three big pages of different kinds
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    sizes = {p: Image.open(p).size for p in pages}   # header only
    pages = [p for p in pages if sizes[p][0] * sizes[p][1] <= 30_000_000]
    big = [p for p in pages if max(sizes[p]) >= LONG]
    picks = [big[0], big[len(big) // 2], big[-1]]
    for k, p in enumerate(picks):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        x1, sx, sy = det_input(g, LONG)
        H, W = x1.shape
        x = np.repeat(x1[None, None], 3, axis=1)
        ref = run("det", "ref", x)["prob"]
        orig = run("det", "orig", x)["prob"]
        cases.append({"model": "det", "id": f"det{k}", "page": p.name, "shape": [1, 3, H, W],
                      "input_gray": save(f"det{k}.x.f32", x1),
                      "ref": {"prob": save(f"det{k}.ref.prob.f32", ref)},
                      "orig": {"prob": save(f"det{k}.orig.prob.f32", orig)}})
        print(f"det{k}: {p.name} {g.shape} -> {H}x{W}, prob {ref.shape}", flush=True)

    # line crops from 10 pages spread over the set
    lines = []
    step = max(1, len(pages) // 10)
    for p in pages[::step][:10]:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        x1, sx, sy = det_input(g, LONG)
        prob = run("det", "orig", np.repeat(x1[None, None], 3, axis=1))["prob"][0, 0]
        bx = boxes_from_prob(prob, sx, sy, g.shape[1], g.shape[0])
        for (x0, y0, x1_, y1) in bx[:32]:
            lines.append(g[y0:y1, x0:x1_])
        if len(lines) >= N_LINES:
            break
    lines = lines[:N_LINES]
    print(f"{len(lines)} line crops", flush=True)

    # router batches of 64
    for b in range(0, len(lines), 64):
        part = lines[b:b + 64]
        x = np.zeros((len(part), 1, 32, 512), np.float32)
        for j, c in enumerate(part):
            l = prep_line(c, 8, 1 << 30)
            idx = np.arange(512) % l.shape[1]
            x[j, 0] = l[:, idx].astype(np.float32) / np.float32(127.5) - np.float32(1.0)
        ref, orig = run("router", "ref", x), run("router", "orig", x)
        cid = f"router{b // 64}"
        cases.append({"model": "router", "id": cid, "shape": list(x.shape), "input": save(f"{cid}.x.f32", x),
                      "ref": {k: save(f"{cid}.ref.{k}.f32", v) for k, v in ref.items()},
                      "orig": {k: save(f"{cid}.orig.{k}.f32", v) for k, v in orig.items()}})

    # recognizer batches: sorted by width, 32 per batch, padded to a multiple of 128 with 1.0
    prepped = [prep_line(c, 8, 1280) for c in lines]
    order = sorted(range(len(prepped)), key=lambda i: prepped[i].shape[1])
    for bi in range(0, len(order), 32):
        idx = order[bi:bi + 32]
        maxw = max(prepped[i].shape[1] for i in idx)
        W = -(-maxw // 128) * 128
        x = np.ones((len(idx), 1, 32, W), np.float32)
        widths = []
        for j, i in enumerate(idx):
            l = prepped[i]
            x[j, 0, :, :l.shape[1]] = l.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
            widths.append(int(l.shape[1]))
        cid = f"rec{bi // 32}"
        xin = save(f"{cid}.x.f32", x)
        for m in MODELS[2:]:
            ref = run(m, "ref", x)
            alt_kind = "noopt" if m == "rec_latin" else "orig"
            alt = run(m, alt_kind, x)
            stride = W // ref["ids"].shape[1]
            cases.append({"model": m, "id": f"{m}.{cid}", "shape": list(x.shape), "input": xin, "widths": widths,
                          "stride": stride,
                          "ref": {"ids": save(f"{m}.{cid}.ref.ids.i32", ref["ids"].astype(np.int32)),
                                  "conf": save(f"{m}.{cid}.ref.conf.f32", ref["conf"])},
                          alt_kind: {"ids": save(f"{m}.{cid}.{alt_kind}.ids.i32", alt["ids"].astype(np.int32)),
                                     "conf": save(f"{m}.{cid}.{alt_kind}.conf.f32", alt["conf"])}})
        print(f"{cid}: {len(idx)} lines, W {W}", flush=True)
    meta = {"onnxruntime": ort.__version__, "pages": str(PAGES), "long_side": LONG, "n_lines": len(lines),
            "levels": {m: json.loads((GPU / "models" / m / "program.json").read_text())["levels"] for m in MODELS},
            "cases": cases}
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
