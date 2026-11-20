#!/usr/bin/env python3
"""NumPy executor for the detector op program (the spec the WGSL follows) and its parity check.

    python3 -I run/ref_det.py                   # both models, every page in data/det_pages/
    python3 -I run/ref_det.py tiny text_two_column

For each model and page: preprocess (the same steps as web/det/engine.js), run the packed
program from web/det/models/<name>/ with NumPy, run onnxruntime on the original ONNX, and
compare the probability maps. Writes:

  data/fixtures/det_<page>/{meta.json, rgba.bin, <model>.ort.f32}   onnxruntime reference for the browser test
  results/det_parity.json                                            NumPy and ORT numbers (WGSL fields added by run_det.mjs)

Preprocessing, shared with engine.js: resize so the long side is LONG (bilinear, half-pixel
centres, clamped), convert RGB to BGR, scale by 1/255, normalise channel c (BGR order) by
mean[c] and std[c] from inference.yml, then zero-pad the bottom and right to multiples of 32.
Zero after normalisation is the mean colour, so the padding is neutral.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "web/det/models"
PAGES_DIR = ROOT / "data/det_pages"
FIX_DIR = ROOT / "data/fixtures"
RESULTS = ROOT / "results/det_parity.json"
ONNX = {"tiny": ROOT / "models/det/tiny/inference.onnx", "small": ROOT / "models/det/small/inference.onnx"}
LONG = 1280
THRESH = 0.2
TOL_NUMPY = 1e-3

# ------------------------------------------------------------------ preprocessing


def bilinear(img, h1, w1):
    """img (h0, w0, 3) float64 -> (h1, w1, 3). Half-pixel centres, clamped to the image."""
    h0, w0 = img.shape[:2]
    ys = np.clip((np.arange(h1) + 0.5) * h0 / h1 - 0.5, 0, h0 - 1)
    xs = np.clip((np.arange(w1) + 0.5) * w0 / w1 - 0.5, 0, w0 - 1)
    y0 = np.floor(ys).astype(np.int64)
    y1 = np.minimum(y0 + 1, h0 - 1)
    x0 = np.floor(xs).astype(np.int64)
    x1 = np.minimum(x0 + 1, w0 - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    rows = img[y0] * (1 - wy) + img[y1] * wy
    return rows[:, x0] * (1 - wx) + rows[:, x1] * wx


def preprocess(rgb, long_side, mean, std):
    """rgb uint8 (h0, w0, 3) -> (x float32 (3, H, W), scale, h1, w1)."""
    h0, w0 = rgb.shape[:2]
    s = long_side / max(h0, w0)
    h1, w1 = max(1, int(math.floor(h0 * s + 0.5))), max(1, int(math.floor(w0 * s + 0.5)))
    img = bilinear(rgb.astype(np.float64), h1, w1)
    bgr = img[..., ::-1] / 255.0
    norm = (bgr - np.array(mean)) / np.array(std)
    H, W = -(-h1 // 32) * 32, -(-w1 // 32) * 32
    x = np.zeros((3, H, W), np.float32)
    x[:, :h1, :w1] = norm.transpose(2, 0, 1)
    return x, s, h1, w1


# ------------------------------------------------------------------ NumPy executor


def erf(x):
    """Abramowitz and Stegun 7.1.26 (max abs error 1.5e-7); engine.js uses the same formula."""
    a = np.abs(x)
    t = np.float32(1) / (np.float32(1) + np.float32(0.3275911) * a)
    y = np.float32(1) - ((((np.float32(1.061405429) * t + np.float32(-1.453152027)) * t + np.float32(1.421413741)) * t
                          + np.float32(-0.284496736)) * t + np.float32(0.254829592)) * t * np.exp(-a * a)
    return np.where(x >= 0, y, -y).astype(np.float32)


def activate(v, act, ha, hb):
    if act == "relu":
        return np.maximum(v, 0)
    if act == "hsig":
        return np.clip(np.float32(ha) * v + np.float32(hb), 0, 1)
    if act == "gelu":
        return (np.float32(0.5) * v * (np.float32(1) + erf(v * np.float32(0.70710678)))).astype(np.float32)
    if act == "sigmoid":
        return (np.float32(1) / (np.float32(1) + np.exp(-v))).astype(np.float32)
    return v


class Weights:
    def __init__(self, raw):
        self.f16 = np.frombuffer(raw, "<f2").astype(np.float32)   # element i = half i, low half first
        self.f32 = np.frombuffer(raw, "<f4")                       # biases, word-addressed

    def conv_w(self, entry):
        """Dequantised weight tensor: hi plane + lo plane (see export_det.Packer.f16_pair)."""
        n = int(np.prod(entry["wshape"]))
        hi = self.f16[2 * entry["w"]: 2 * entry["w"] + n]
        lo = self.f16[2 * entry["w"] + entry["wl"]: 2 * entry["w"] + entry["wl"] + n]
        return (hi + lo).reshape(entry["wshape"])

    def bias(self, entry, n):
        return self.f32[entry["b"]: entry["b"] + n]


def conv(x, e, W_, B_):
    """Conv with explicit pads [top, left, bottom, right], dilation, groups; x (C, H, W)."""
    kh, kw = e["k"]
    sh, sw = e["s"]
    dh, dw = e["d"]
    pt, pl, pb, pr = e["pads"]
    g = e["g"]
    O, icg = W_.shape[0], W_.shape[1]
    C, H, Wd = x.shape
    oh = (H + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (Wd + pl + pr - dw * (kw - 1) - 1) // sw + 1
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)))
    out = np.broadcast_to(B_[:, None, None], (O, oh, ow)).astype(np.float32)
    ocg = O // g
    for gi in range(g):
        xs = xp[gi * icg:(gi + 1) * icg]
        for ky in range(kh):
            for kx in range(kw):
                patch = xs[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw]
                wk = W_[gi * ocg:(gi + 1) * ocg, :, ky, kx]
                out[gi * ocg:(gi + 1) * ocg] += np.tensordot(wk, patch, axes=([1], [0])).astype(np.float32)
    return out


def convt2(x, W_, B_):
    """ConvTranspose with kernel 2, stride 2, no pads: each output pixel has one tap."""
    Cin, O, _, _ = W_.shape
    _, H, Wd = x.shape
    out = np.zeros((O, 2 * H, 2 * Wd), np.float32)
    for ky in range(2):
        for kx in range(2):
            out[:, ky::2, kx::2] = np.einsum("chw,co->ohw", x, W_[:, :, ky, kx]).astype(np.float32)
    return out + B_[:, None, None]


def maxpool(x, e):
    kh, kw = e["k"]
    sh, sw = e["s"]
    pt, pl, pb, pr = e["pads"]
    C, H, Wd = x.shape
    oh = (H + pt + pb - kh) // sh + 1
    ow = (Wd + pl + pr - kw) // sw + 1
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    out = np.full((C, oh, ow), -np.inf, np.float32)
    for ky in range(kh):
        for kx in range(kw):
            out = np.maximum(out, xp[:, ky: ky + sh * (oh - 1) + 1: sh, kx: kx + sw * (ow - 1) + 1: sw])
    return out


def run_program(man, wts, x, trace=None):
    """Execute the op program on x (3, H, W). Returns the probability map (H, W).
    trace, if a dict, receives every op's output buffer keyed by op index (debugging)."""
    bufs = {0: x}
    for oi, op in enumerate(man["ops"]):
        k = op["op"]
        if k == "conv":
            W_ = wts.conv_w(op)
            B_ = wts.bias(op, W_.shape[0])
            bufs[op["out"]] = activate(conv(bufs[op["in"]], op, W_, B_), op["act"], op["ha"], op["hb"])
        elif k == "convt":
            W_ = wts.conv_w(op)
            B_ = wts.bias(op, W_.shape[1])
            bufs[op["out"]] = activate(convt2(bufs[op["in"]], W_, B_), op["act"], op["ha"], op["hb"])
        elif k == "se":
            v = bufs[op["in"]]
            m = v.reshape(v.shape[0], 1, -1).mean(axis=2, dtype=np.float32)[:, :, None]
            w1, w2 = wts.conv_w(op["fc1"]), wts.conv_w(op["fc2"])
            h = activate(conv(m, op["fc1"], w1, wts.bias(op["fc1"], w1.shape[0])), "relu", 0, 0)
            g = activate(conv(h, op["fc2"], w2, wts.bias(op["fc2"], w2.shape[0])), "hsig",
                         op["fc2"]["ha"], op["fc2"]["hb"])
            y = v * g
            bufs[op["out"]] = (y + v if op["res"] else y).astype(np.float32)
        elif k == "maxpool":
            bufs[op["out"]] = maxpool(bufs[op["in"]], op)
        elif k == "concat":
            bufs[op["out"]] = np.concatenate([bufs[i] for i in op["ins"]], axis=0)
        elif k == "resize":
            s = op["scale"]
            bufs[op["out"]] = np.repeat(np.repeat(bufs[op["in"]], s, axis=1), s, axis=2)
        elif k == "add":
            bufs[op["out"]] = (bufs[op["ins"][0]] + bufs[op["ins"][1]]).astype(np.float32)
        else:
            raise ValueError(f"unknown op {k}")
        if trace is not None:
            trace[oi] = bufs[op["out"]]
    return bufs[man["output"]["buf"]][0]


def load_model(name):
    d = MODELS_DIR / name
    man = json.loads((d / "manifest.json").read_text())
    return man, Weights((d / man["weights"]["file"]).read_bytes())


# ------------------------------------------------------------------ ORT reference and checks


def ort_session(name, level="basic"):
    """Reference session. "basic" keeps the ONNX graph's arithmetic (identical to unoptimised here).
    "all" lets onnxruntime fuse the GELU chains into its own kernels; that drifts by about 1.8e-3
    on heading_table, so it is recorded for the record but is not the reference."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if level == "basic"
                                   else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    return ort.InferenceSession(str(ONNX[name]), so, providers=["CPUExecutionProvider"])


def ort_run(sess, x):
    t0 = time.perf_counter()
    out = sess.run(None, {"x": x[None]})[0]
    return out[0, 0], (time.perf_counter() - t0) * 1000


def mismatch(a, b):
    return float(np.mean((a > THRESH) != (b > THRESH)))


def load_page(name):
    im = Image.open(PAGES_DIR / f"{name}.png").convert("RGB")
    return np.asarray(im)


def write_fixture(name, rgb):
    d = FIX_DIR / f"det_{name}"
    d.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
    (d / "rgba.bin").write_bytes(rgba.tobytes())
    (d / "meta.json").write_text(json.dumps({"name": name, "w": w, "h": h, "long_side": LONG}) + "\n")
    return d


def main(argv):
    names = [a for a in argv if a in ONNX] or list(ONNX)
    pages = [a for a in argv if a not in ONNX] or sorted(p.stem for p in PAGES_DIR.glob("*.png"))
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.update({"task": "PP-OCRv6 det port: NumPy executor and WGSL vs onnxruntime",
                    "long_side": LONG, "threshold": THRESH,
                    "tolerance": {"numpy_max_abs": TOL_NUMPY, "wgsl_max_abs": 2e-3, "mismatch_share_max": 1e-3}})
    results.setdefault("models", {})
    ok = True
    for name in names:
        man, wts = load_model(name)
        sess = ort_session(name)
        sess_all = ort_session(name, "all")
        mean, std = man["input"]["mean"], man["input"]["std"]
        results["models"].setdefault(name, {"pages": {}})
        for page in pages:
            rgb = load_page(page)
            x, scale, h1, w1 = preprocess(rgb, LONG, mean, std)
            ref, ort_ms = ort_run(sess, x)
            ref_all, _ = ort_run(sess_all, x)
            t0 = time.perf_counter()
            got = run_program(man, wts, x)
            np_ms = (time.perf_counter() - t0) * 1000
            d = float(np.max(np.abs(got - ref)))
            mm = mismatch(got, ref)
            fx = write_fixture(page, rgb)
            ref.astype("<f4").tofile(fx / f"{name}.ort.f32")
            rec = results["models"][name]["pages"].setdefault(page, {})
            rec.update({"H": int(x.shape[1]), "W": int(x.shape[2]), "valid_h": h1, "valid_w": w1,
                        "scale": round(scale, 6), "numpy_max_abs": d, "numpy_mismatch_share": mm,
                        "numpy_ms": round(np_ms), "ort_ms": round(ort_ms),
                        "ort_all_opt_max_abs_vs_ref": float(np.max(np.abs(ref_all - ref)))})
            ok = ok and d <= TOL_NUMPY
            print(f"{name:5s} {page:18s} {x.shape[1]}x{x.shape[2]}  numpy max|d|={d:.2e}  "
                  f"mismatch@0.2={mm:.2e}  numpy {np_ms/1000:.1f}s  ort {ort_ms:.0f}ms", flush=True)
    results["numpy_pass"] = ok
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print("numpy pass" if ok else "numpy FAIL", "->", RESULTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
