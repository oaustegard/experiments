"""NumPy forward passes for the packed models: the spec the browser engine
follows, checked against ONNX Runtime on the matching fake-quantized arm.

    python3 run/ref.py PACK_DIR ARM_DIR     # parity of every model on random inputs
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ pack loader

def load_pack(d: Path, name: str) -> tuple[dict, dict]:
    man = json.loads((d / "manifest.json").read_text())
    e = man["models"][name]
    buf = (d / e["file"]).read_bytes()
    out = {}
    for tn, r in e["tensors"].items():
        shape = r["shape"]
        n = int(np.prod(shape))
        if r["enc"] == "f16":
            out[tn] = np.frombuffer(buf, np.float16, n, r["data"]).astype(np.float32).reshape(shape)
        elif r["enc"] == "f32":
            out[tn] = np.frombuffer(buf, np.float32, n, r["data"]).reshape(shape)
        else:
            O, K, G = r["O"], r["K"], r["G"]
            nb = (O * K * r["bits"] + 7) // 8
            bitsarr = np.unpackbits(np.frombuffer(buf, np.uint8, nb, r["codes"]), bitorder="little")[: O * K * r["bits"]]
            codes = (bitsarr.reshape(O * K, r["bits"]).astype(np.int64) << np.arange(r["bits"])).sum(1)
            codes = codes.astype(np.float32).reshape(O, K)
            rows = O * (-(-K // G)) if G else O
            sc = np.frombuffer(buf, np.float16, rows, r["scales"]).astype(np.float32)
            if r["sym"]:
                codes = codes - (2 ** (r["bits"] - 1) - 1)
            if G:
                ng = -(-K // G)
                sc = np.repeat(sc.reshape(O, ng), G, axis=1)[:, :K]
                if not r["sym"]:
                    zp = np.frombuffer(buf, np.uint8, rows, r["zp"]).astype(np.float32)
                    codes = codes - np.repeat(zp.reshape(O, ng), G, axis=1)[:, :K]
                w2 = codes * sc
            else:
                if not r["sym"]:
                    codes = codes - np.frombuffer(buf, np.uint8, O, r["zp"]).astype(np.float32)[:, None]
                w2 = codes * sc[:, None]
            # [O, K] back to the ONNX shape
            out[tn] = w2.reshape(shape) if r["view"] == "conv" else w2.T.reshape(shape)
    return e, out


# ------------------------------------------------------------------ ops (NHWC)

def hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6


ACT = {None: lambda x: x, "hardswish": hardswish, "relu": lambda x: np.maximum(x, 0),
       "sigmoid": lambda x: 1 / (1 + np.exp(-x))}


def conv(x, w, b, k, s, p, g):
    """x [N, H, W, C]; w ONNX [O, C/g, kh, kw]; pads [top, left, bottom, right]."""
    N, H, W, C = x.shape
    O = w.shape[0]
    kh, kw = k
    xp = np.pad(x, ((0, 0), (p[0], p[2]), (p[1], p[3]), (0, 0)))
    Ho = (H + p[0] + p[2] - kh) // s[0] + 1
    Wo = (W + p[1] + p[3] - kw) // s[1] + 1
    cg, og = C // g, O // g
    y = np.zeros((N, Ho, Wo, O), np.float32)
    for gi in range(g):
        xs = xp[..., gi * cg:(gi + 1) * cg]
        wg = w[gi * og:(gi + 1) * og]  # [og, cg, kh, kw]
        for dy in range(kh):
            for dx in range(kw):
                patch = xs[:, dy:dy + s[0] * (Ho - 1) + 1:s[0], dx:dx + s[1] * (Wo - 1) + 1:s[1], :]
                y[..., gi * og:(gi + 1) * og] += patch @ wg[:, :, dy, dx].T
    if b is not None:
        y += b
    return y


def run_convnet(e, W, x):
    """x [N, H, W, C] -> every tensor of the program."""
    t = {e["input"]: x}
    for op in e["ops"]:
        o = op["op"]
        if o == "conv":
            y = conv(t[op["in"]], W[op["w"]], W.get(op["b"]), op["k"], op["s"], op["p"], op["g"])
            if op["res"] and op["res_at"] == "pre":
                y = y + t[op["res"]]
            y = ACT[op["act"]](y)
            if op["res"] and op["res_at"] == "post":
                y = y + t[op["res"]]
        elif o == "act":
            y = ACT[op["act"]](t[op["in"]])
        elif o == "add":
            y = t[op["a"]] + t[op["b"]]
        elif o == "up2":
            y = t[op["in"]].repeat(2, 1).repeat(2, 2)
        elif o == "s2d":
            a = t[op["in"]]
            N, H, Wd, C = a.shape
            # out channel c*4 + dy*2 + dx
            y = a.reshape(N, H // 2, 2, Wd // 2, 2, C).transpose(0, 1, 3, 5, 2, 4).reshape(N, H // 2, Wd // 2, C * 4)
        elif o == "concat":
            y = np.concatenate([t[i] for i in op["ins"]], -1)
        elif o == "d2s":
            a = t[op["in"]]
            N, H, Wd, C4 = a.shape
            bs = op["bs"]
            assert op["mode"] == "CRD"
            y = a.reshape(N, H, Wd, C4 // (bs * bs), bs, bs).transpose(0, 1, 4, 2, 5, 3).reshape(N, H * bs, Wd * bs, C4 // (bs * bs))
        elif o == "pool":
            a = t[op["in"]]
            N, H, Wd, C = a.shape
            a = a[:, : H // 2 * 2, : Wd // 2 * 2].reshape(N, H // 2, 2, Wd // 2, 2, C)
            y = a.mean((2, 4)) if op["kind"] == "avg" else a.max((2, 4))
        elif o == "gmean":
            y = t[op["in"]].mean((1, 2))
        elif o == "hmean":
            y = t[op["in"]].mean(1)  # [N, W, C]
        elif o == "gemm":
            y = ACT[op["act"]](t[op["in"]] @ W[op["w"]].T + W[op["b"]])
        t[op["out"]] = y
    return t


def det(e, W, x):
    """x [H, W] ink map -> [3, H/2, W/2]: core logit, top and bottom distance (ONNX det's out)."""
    t = run_convnet(e, W, x[None, :, :, None])
    y = t[e["ops"][-1]["out"]][0]  # [H/2, W/2, 3]
    sp = np.log1p(np.exp(-np.abs(y[..., 1:]))) + np.maximum(y[..., 1:], 0)
    return np.concatenate([y[..., :1], sp * e["post"]["softplus_scale"]], -1).transpose(2, 0, 1)


def classify(e, W, x, widths=None):
    """orient: x [N, 768, 768] -> softmax [N, 5]; script: x [N, 32, W], widths -> softmax [N, 4]."""
    t = run_convnet(e, W, x[..., None])
    last = e["ops"][-1]
    pre = [op for op in e["ops"] if op["op"] == "gemm"][0]
    feat = t[pre["in"]]
    if pre.get("wdiv"):  # script: [N, W', C] masked mean over valid columns
        n = np.maximum(widths // pre["wdiv"], 1)
        m = (np.arange(feat.shape[1])[None, :] < n[:, None]).astype(np.float32)
        feat = (feat * m[..., None]).sum(1) / m.sum(1, keepdims=True)
    z = feat @ W[pre["w"]].T + W[pre["b"]]
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


# ------------------------------------------------------------------ recognizer

def _dwname(W, i):
    for k in (f"rec.blocks.{i}.dw.Conv.weight", f"rec.blocks.{i}.dw.weight"):
        if k in W:
            return k, k[:-6] + "bias"
    raise KeyError(i)


def rec(e, W, x, widths):
    """x [B, H, Wd] (H = 96 for the windowed base recognizer, 32 otherwise), widths [B]
    -> logits [B, T, C] and lengths [B]."""
    B, H, Wd = x.shape
    win = "rec.context.row_prior" in W
    band = x[:, 32:64] if win else x
    stem = conv(band[..., None], W["rec.stem.weight"], W["rec.stem.bias"], [32, 4], [32, 4], [0, 0, 0, 0], 1)[:, 0]  # [B, T, 128]
    T = stem.shape[1]
    if win:
        c = conv(x[..., None], W["rec.context.patch.Conv.weight"], W["rec.context.patch.Conv.bias"], [4, 4], [4, 4], [0] * 4, 1)
        c = hardswish(c)
        c = conv(c, W["rec.context.dw.weight"], None, [3, 3], [1, 1], [1] * 4, 32)
        c = hardswish(conv(c, W["rec.context.pw.Conv.weight"], W["rec.context.pw.Conv.bias"], [1, 1], [1, 1], [0] * 4, 1))  # [B, 24, T, 64]
        sc = conv(c, W["rec.context.score.weight"], W["rec.context.score.bias"], [1, 1], [1, 1], [0] * 4, 1)  # [B, 24, T, 4]
        sc = sc + W["rec.context.row_prior"][0, :, :, 0].T[None, :, None, :]
        sc = np.exp(sc - sc.max(1, keepdims=True))
        sc = sc / sc.sum(1, keepdims=True)
        # channels [4 heads x 16] weighted by their head's row softmax, summed over rows
        cc = c.reshape(B, 24, T, 4, 16) * sc[..., None]
        ctx = cc.sum(1).reshape(B, T, 64)
        stem = stem + ctx @ W["rec.context.proj.weight"][:, :, 0].T + W["rec.context.proj.bias"]
    lengths = np.minimum((widths + 3) // 4, T)
    mask = (np.arange(T)[None, :] < lengths[:, None]).astype(np.float32)[..., None]
    g, bb, mu, var = (W[f"rec.stem_bn.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))
    h = (stem - mu) / np.sqrt(var + 1e-5) * g + bb
    h = hardswish(h) * mask
    cnt = np.maximum(mask.sum(1, keepdims=True), 1.0)
    for i in range(10):
        dw, dwb = _dwname(W, i)
        y = conv(h[:, None], W[dw][:, :, None, :], W[dwb], [1, 9], [1, 1], [0, 4, 0, 4], 128)[:, 0]
        y = hardswish(y @ W[f"rec.blocks.{i}.expand.weight"][:, :, 0].T + W[f"rec.blocks.{i}.expand.bias"])
        s = (y * mask).sum(1, keepdims=True) / cnt
        s = np.maximum(s @ W[f"rec.blocks.{i}.se.fc1.weight"][:, :, 0].T + W[f"rec.blocks.{i}.se.fc1.bias"], 0)
        s = 1 / (1 + np.exp(-(s @ W[f"rec.blocks.{i}.se.fc2.weight"][:, :, 0].T + W[f"rec.blocks.{i}.se.fc2.bias"])))
        y = (y * s) @ W[f"rec.blocks.{i}.project.weight"][:, :, 0].T + W[f"rec.blocks.{i}.project.bias"]
        h = (h + y) * mask
    logits = h @ W["rec.head.weight"][:, :, 0].T + W["rec.head.bias"]
    return logits, lengths


def ctc_outputs(logits):
    """The recognizer's ONNX outputs from its logits: ids, conf, blank, topv, topi."""
    m = logits.max(-1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(-1)
    lsm = logits - m - np.log(s)[..., None]
    topi = np.argsort(-lsm, -1, kind="stable")[..., :8]
    return {"ids": logits.argmax(-1), "conf": 1.0 / s, "blank": lsm[..., 0],
            "topv": np.take_along_axis(lsm, topi, -1), "topi": topi}


# ------------------------------------------------------------------ order model

def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))


FREQS = (np.pi / 2) * 2.0 ** np.arange(16)


def order(e, W, feats):
    """feats [N, 16] -> a [N, N], cls [N, 10], same [N, N], tab [3, N, N]."""
    N = len(feats)
    f = feats[:, :4, None] * FREQS.astype(np.float32)
    four = np.concatenate([np.sin(f), np.cos(f)], -1).reshape(N, 128)
    x = np.concatenate([feats, four], -1)
    x = _gelu(x @ W["inp.inp.0.MatMul.weight"] + W["inp.0.bias"])
    x = x @ W["inp.inp.2.MatMul.weight"] + W["inp.2.bias"]
    for i in range(4):
        L = f"layers.{i}"
        h = _ln(x, W[f"{L}.n1.weight"], W[f"{L}.n1.bias"])
        qkv = (h @ W[f"{L}.qkv.MatMul.weight"] + W[f"{L}.qkv.bias"]).reshape(N, 3, 4, 32).transpose(1, 2, 0, 3)
        q, k, v = qkv
        s = q @ k.transpose(0, 2, 1) / math.sqrt(32)
        s = np.exp(s - s.max(-1, keepdims=True))
        s = s / s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(N, 128)
        x = x + o @ W[f"{L}.proj.MatMul.weight"] + W[f"{L}.proj.bias"]
        h = _ln(x, W[f"{L}.n2.weight"], W[f"{L}.n2.bias"])
        h = _gelu(h @ W[f"{L}.ff.ff.0.MatMul.weight"] + W[f"{L}.ff.0.bias"])
        x = x + h @ W[f"{L}.ff.ff.2.MatMul.weight"] + W[f"{L}.ff.2.bias"]
    x = _ln(x, W["norm.weight"], W["norm.bias"])
    r = math.sqrt(128)

    def lin(n):
        return x @ W[f"{n}.MatMul.weight"] + W[f"{n}.bias"]

    s = lin("q") @ lin("k").T / r
    a = s - s.T
    s2 = lin("bu") @ lin("bw").T / r
    same = s2 + s2.T
    tu, tw = lin("tu").reshape(N, 3, 128).transpose(1, 0, 2), lin("tw").reshape(N, 3, 128).transpose(1, 2, 0)
    t = tu @ tw / r
    tab = t + t.transpose(0, 2, 1)
    return a, lin("cls"), same, tab


# ------------------------------------------------------------------ parity check

def main():
    import onnxruntime as ort

    pack, arm = Path(sys.argv[1]), Path(sys.argv[2])
    rng = np.random.default_rng(0)

    def sess(n):
        return ort.InferenceSession(str(arm / f"{n}.onnx"), providers=["CPUExecutionProvider"])

    def report(name, got, want):
        d = float(np.abs(got - want).max())
        print(f"{name:12s} max|diff| {d:.2e}  (max|ref| {float(np.abs(want).max()):.2e})")

    x = (rng.random((128, 96)) < 0.2).astype(np.float32)
    e, W = load_pack(pack, "det")
    report("det", det(e, W, x), sess("det").run(None, {"x": x[None, None]})[0][0])
    x = (rng.random((2, 768, 768)) < 0.05).astype(np.float32)
    e, W = load_pack(pack, "orient")
    report("orient", classify(e, W, x), sess("orient").run(None, {"x": x[:, None]})[0])
    x = (rng.random((3, 32, 200)) < 0.2).astype(np.float32)
    wd = np.array([200, 130, 40])
    e, W = load_pack(pack, "script")
    report("script", classify(e, W, x, wd), sess("script").run(None, {"x": x[:, None], "widths": wd})[0])
    for name, H in (("rec", 96), ("rec_world", 32)):
        x = (rng.random((3, H, 160)) < 0.2).astype(np.float32)
        wd = np.array([160, 101, 33])
        e, W = load_pack(pack, name)
        lg, ln = rec(e, W, x, wd)
        got = ctc_outputs(lg)
        want = dict(zip(["ids", "conf", "lengths", "blank", "topv", "topi"],
                        sess(name).run(None, {"x": x[:, None], "widths": wd})))
        assert (want["lengths"] == ln).all(), (want["lengths"], ln)
        for k in ("conf", "blank", "topv"):
            for b in range(3):
                report(f"{name}.{k}[{b}]", got[k][b, : ln[b]], want[k][b, : ln[b]])
        print(f"{name:12s} ids equal: {all((got['ids'][b,:ln[b]] == want['ids'][b,:ln[b]]).all() for b in range(3))}")
    feats = rng.random((9, 16)).astype(np.float32)
    e, W = load_pack(pack, "order")
    got = order(e, W, feats)
    want = sess("order").run(None, {"feats": feats[None], "mask": np.ones((1, 9), bool)})
    for nm, g, w in zip(["a", "cls", "same", "tab"], got, want):
        report(f"order.{nm}", g, w[0])


if __name__ == "__main__":
    main()
