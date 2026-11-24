#!/usr/bin/env python3
"""NumPy executor for the recognizer op program (the spec web/rec/engine.js follows) and its parity check.

    python3 -I web/rec/test/make_lines.py      # synthetic line crops -> web/rec/test/fixtures/
    python3 -I run/ref_rec.py                  # both models, every fixture line
    python3 -I run/ref_rec.py tiny --pack r63  # one model; --pack names web/rec/models/<name>[-<pack>]

For each model and line: run the packed program from web/rec/models/<name>/ (default scheme r63) and
from web/rec/models/<name>-f32/ (unquantized) with NumPy, run onnxruntime on the original ONNX
(ORT_ENABLE_BASIC: at full optimisation onnxruntime fuses the GELU chains into its own kernels), and
compare the [T, V] probabilities and the greedy CTC text. Writes, for the browser test:

  web/rec/test/fixtures/<line>/<model>.ref.f32   NumPy probabilities with the packed weights
  web/rec/test/fixtures/<line>/<model>.ort.f32   onnxruntime probabilities (fp32 ONNX)
  web/rec/test/fixtures/batch4/<model>.ref.f32   NumPy probabilities of the padded 4-line batch

and results/rec_parity.json (NumPy numbers; web/rec/test/run_rec.mjs adds the WGSL ones). The bar
here: NumPy with f32 weights matches onnxruntime within 1e-4 max abs on every line. Exit 0 when it does.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "run"))
from export_rec import dequant  # noqa: E402

MODELS = ROOT / "web/rec/models"
FIX = ROOT / "web/rec/test/fixtures"
SRC = ROOT / "models"
RESULTS = ROOT / "results/rec_parity.json"
TOL_ORT = 1e-4

# ------------------------------------------------------------------ elementwise


def erf(x):
    """Abramowitz and Stegun 7.1.26 (max abs error 1.5e-7); kernels.js uses the same formula."""
    a = np.abs(x)
    t = np.float32(1) / (np.float32(1) + np.float32(0.3275911) * a)
    y = np.float32(1) - ((((np.float32(1.061405429) * t + np.float32(-1.453152027)) * t + np.float32(1.421413741)) * t
                          + np.float32(-0.284496736)) * t + np.float32(0.254829592)) * t * np.exp(-a * a)
    return np.where(x >= 0, y, -y).astype(np.float32)


def sigmoid(v):
    return (np.float32(1) / (np.float32(1) + np.exp(-v))).astype(np.float32)


def activate(v, act, ha, hb):
    if act == "relu":
        return np.maximum(v, np.float32(0))
    if act == "hsig":
        return np.clip(np.float32(ha) * v + np.float32(hb), 0, 1).astype(np.float32)
    if act == "gelu":
        return (np.float32(0.5) * v * (np.float32(1) + erf(v * np.float32(0.70710678)))).astype(np.float32)
    if act == "sigmoid":
        return sigmoid(v)
    if act == "hswish":
        return (v * np.clip(np.float32(ha) * v + np.float32(hb), 0, 1)).astype(np.float32)
    if act == "swish":
        return (v * sigmoid(v)).astype(np.float32)
    assert act == "none", act
    return v


# ------------------------------------------------------------------ ops (one line: chw = (C, H, W), tc = (T, C))


def conv(x, e, W_, B_):
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
    if g == C and icg == 1 and ocg == 1:   # depthwise
        for ky in range(kh):
            for kx in range(kw):
                patch = xp[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw]
                out = out + W_[:, 0, ky, kx][:, None, None] * patch
        return activate(out.astype(np.float32), e["act"], e["ha"], e["hb"])
    for gi in range(g):
        xs = xp[gi * icg:(gi + 1) * icg]
        for ky in range(kh):
            for kx in range(kw):
                patch = xs[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw]
                wk = W_[gi * ocg:(gi + 1) * ocg, :, ky, kx]
                out[gi * ocg:(gi + 1) * ocg] += np.tensordot(wk, patch, axes=([1], [0])).astype(np.float32)
    return activate(out, e["act"], e["ha"], e["hb"])


def pool(x, e, kind):
    kh, kw = e["k"]
    sh, sw = e["s"]
    pt, pl, pb, pr = e["pads"]
    C, H, Wd = x.shape
    oh = (H + pt + pb - kh) // sh + 1
    ow = (Wd + pl + pr - kw) // sw + 1
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr)), constant_values=fill)
    ones = np.pad(np.ones((1, H, Wd), np.float32), ((0, 0), (pt, pb), (pl, pr)))
    acc = np.full((C, oh, ow), -np.inf if kind == "max" else 0.0, np.float32)
    cnt = np.zeros((1, oh, ow), np.float32)
    for ky in range(kh):
        for kx in range(kw):
            sl = (slice(None), slice(ky, ky + sh * (oh - 1) + 1, sh), slice(kx, kx + sw * (ow - 1) + 1, sw))
            if kind == "max":
                acc = np.maximum(acc, xp[sl])
            else:
                acc = acc + xp[sl]
                cnt = cnt + ones[sl]
    if kind == "max":
        return acc
    div = np.float32(kh * kw) if e["cip"] else cnt
    return (acc / div).astype(np.float32)


def se(x, op, Wt):
    m = x.reshape(x.shape[0], -1).astype(np.float64).mean(axis=1).astype(np.float32)[:, None, None]
    h = conv(m, op["fc1"], Wt[op["fc1"]["w"]], Wt[op["fc1"]["b"]])
    gate = conv(h, op["fc2"], Wt[op["fc2"]["w"]], Wt[op["fc2"]["b"]])
    y = x * gate
    return (y + x) if op["res"] else y


def layernorm(x, g, b, eps):
    mean = x.mean(axis=-1, keepdims=True, dtype=np.float32)
    d = x - mean
    var = (d * d).mean(axis=-1, keepdims=True, dtype=np.float32)
    return (d / np.sqrt(var + np.float32(eps)) * g + b).astype(np.float32)


def attention(qkv, nh, hd, scale):
    T = qkv.shape[0]
    D = nh * hd
    q = qkv[:, :D].reshape(T, nh, hd).transpose(1, 0, 2) * np.float32(scale)
    k = qkv[:, D:2 * D].reshape(T, nh, hd).transpose(1, 0, 2)
    v = qkv[:, 2 * D:].reshape(T, nh, hd).transpose(1, 0, 2)
    s = q @ k.transpose(0, 2, 1)
    p = softmax(s)
    return (p @ v).transpose(1, 0, 2).reshape(T, D).astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class Program:
    """A packed program from web/rec/models/<dir>/: forward(x (3, 48, W)) -> probs (T, V)."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        raw = (self.dir / "weights.bin").read_bytes()
        assert len(raw) == self.man["weights"]["bytes"]
        self.W = [dequant(e, raw) for e in self.man["tensors"]]
        chars = (self.dir / self.man["dict"]["file"]).read_text(encoding="utf-8").split("\n")
        if chars and chars[-1] == "":
            chars.pop()
        assert len(chars) == self.man["dict"]["entries"]
        self.chars = ["<blank>"] + chars + [" "]
        assert len(self.chars) == self.man["V"]

    def forward(self, x):
        man, Wt = self.man, self.W
        bufs = {0: np.asarray(x, np.float32)}
        last = {}
        for i, op in enumerate(man["ops"]):
            for b in op.get("ins", [op.get("in")]):
                last[b] = i
        for i, op in enumerate(man["ops"]):
            kind = op["op"]
            if kind == "conv":
                y = conv(bufs[op["in"]], op, Wt[op["w"]], Wt[op["b"]])
            elif kind == "se":
                y = se(bufs[op["in"]], op, Wt)
            elif kind == "add":
                y = bufs[op["ins"][0]] + bufs[op["ins"][1]]
            elif kind == "concat":
                y = np.concatenate([bufs[b] for b in op["ins"]], axis=0)
            elif kind == "avgpool":
                y = pool(bufs[op["in"]], op, "avg")
            elif kind == "maxpool":
                y = pool(bufs[op["in"]], op, "max")
            elif kind == "img2seq":
                a = bufs[op["in"]]
                y = np.ascontiguousarray(a.reshape(a.shape[0], -1).T)
            elif kind == "seq2img":
                a = bufs[op["in"]]
                y = np.ascontiguousarray(a.T)[:, None, :]
            elif kind == "linear":
                y = activate((bufs[op["in"]] @ Wt[op["w"]] + Wt[op["b"]]).astype(np.float32), op["act"], op["ha"], op["hb"])
            elif kind == "ln":
                y = layernorm(bufs[op["in"]], Wt[op["g"]], Wt[op["b"]], op["eps"])
            elif kind == "attn":
                y = attention(bufs[op["in"]], op["nh"], op["hd"], op["scale"])
            elif kind == "softmax":
                y = softmax(bufs[op["in"]])
            else:
                raise ValueError(kind)
            bufs[op["out"]] = y.astype(np.float32)
            for b in op.get("ins", [op.get("in")]):
                if last.get(b) == i and b != man["output"]["buf"]:
                    bufs.pop(b, None)
        return bufs[man["output"]["buf"]]

    def decode(self, probs):
        return greedy(probs, self.chars)


def greedy(probs, chars):
    """PaddleOCR CTCLabelDecode: argmax per step, merge repeats, drop blank (0)."""
    idx = probs.argmax(axis=-1)
    keep = np.ones(len(idx), bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0
    return "".join(chars[i] for i in idx[keep])


def ort_session(name):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(SRC / f"{name}_rec/inference.onnx"), so, providers=["CPUExecutionProvider"])


def cer(a, b):
    """Levenshtein distance / len(b)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*", default=["tiny", "small"])
    ap.add_argument("--pack", default="", help="packed variant: web/rec/models/<name>-<pack> (default: <name>)")
    a = ap.parse_args(argv)
    index = json.loads((FIX / "index.json").read_text())
    lines = index["lines"]
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res.setdefault("models", {})
    res["fixtures"] = {"lines": len(lines), "widths": [e["W"] for e in lines], "batch": index["batch"]}
    ok = True
    for name in a.models:
        pq = Program(MODELS / (f"{name}-{a.pack}" if a.pack else name))
        pf = Program(MODELS / f"{name}-f32")
        sess = ort_session(name)
        iname = sess.get_inputs()[0].name
        V = pq.man["V"]
        rows = {}
        t0 = time.time()
        for e in lines:
            d = FIX / e["id"]
            x = np.fromfile(d / "x.f32", np.float32).reshape(3, 48, e["W"])
            p_ort = sess.run(None, {iname: x[None]})[0][0]
            p_f32 = pf.forward(x)
            p_q = pq.forward(x)
            assert p_ort.shape == p_q.shape == p_f32.shape and p_q.shape[1] == V
            p_q.astype("<f4").tofile(d / f"{name}.ref.f32")
            p_ort.astype("<f4").tofile(d / f"{name}.ort.f32")
            t_ort, t_q = greedy(p_ort, pq.chars), pq.decode(p_q)
            r = {
                "W": e["W"], "T": int(p_q.shape[0]),
                "numpy_f32_vs_ort_max_abs": float(np.abs(p_f32 - p_ort).max()),
                "numpy_packed_vs_ort_max_abs": float(np.abs(p_q - p_ort).max()),
                "text_ort": t_ort, "text_packed": t_q, "text_f32": pf.decode(p_f32),
                "gt": e["text"], "cer_ort": round(cer(t_ort, e["text"]), 4), "cer_packed": round(cer(t_q, e["text"]), 4),
            }
            r["numpy_pass"] = r["numpy_f32_vs_ort_max_abs"] <= TOL_ORT and r["text_f32"] == t_ort
            ok = ok and r["numpy_pass"]
            rows[e["id"]] = r
            print(f"{name} {e['id']:9s} W={e['W']:4d} T={r['T']:3d} f32-ort {r['numpy_f32_vs_ort_max_abs']:.2e} "
                  f"packed-ort {r['numpy_packed_vs_ort_max_abs']:.2e} | {t_q!r}" + ("" if t_q == t_ort else f" (ort {t_ort!r})"),
                  flush=True)
        # padded batch: every line of the batch through the packed program, as the engine sees it
        b = index["batch"]
        xb = np.fromfile(FIX / b["id"] / "x.f32", np.float32).reshape(b["B"], 3, 48, b["W"])
        pb = np.stack([pq.forward(xb[j]) for j in range(b["B"])])
        pb.astype("<f4").tofile(FIX / b["id"] / f"{name}.ref.f32")
        m = res["models"].setdefault(name, {})
        m.update({
            "pack": pq.dir.name, "V": V, "lines": rows,
            "numpy_f32_vs_ort_max_abs": max(r["numpy_f32_vs_ort_max_abs"] for r in rows.values()),
            "numpy_packed_vs_ort_max_abs": max(r["numpy_packed_vs_ort_max_abs"] for r in rows.values()),
            "packed_text_same_as_ort": sum(r["text_packed"] == r["text_ort"] for r in rows.values()),
            "mean_cer_ort": round(float(np.mean([r["cer_ort"] for r in rows.values()])), 4),
            "mean_cer_packed": round(float(np.mean([r["cer_packed"] for r in rows.values()])), 4),
            "batch_texts_packed": [pq.decode(pb[j]) for j in range(b["B"])],
            "numpy_pass": all(r["numpy_pass"] for r in rows.values()),
            "numpy_seconds": round(time.time() - t0, 1),
        })
        print(f"{name}: f32 vs ort max {m['numpy_f32_vs_ort_max_abs']:.2e}; packed ({pq.dir.name}) vs ort max "
              f"{m['numpy_packed_vs_ort_max_abs']:.2e}, same text {m['packed_text_same_as_ort']}/{len(rows)}; "
              f"CER ort {m['mean_cer_ort']} packed {m['mean_cer_packed']}")
    res["numpy_pass"] = all(m.get("numpy_pass", False) for m in res["models"].values())
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
