"""Weight quantizers for the packed Indra models, and fake-quantized ONNX arms.

A weight is viewed as [O, K]: O output channels (Conv axis 0, MatMul axis 1,
Gemm by transB), K the rest of its axes flattened, in the order the runtime
reads them. A scheme rounds it to codes and scales; `fake` turns the codes
back into floats, so ONNX Runtime with those floats computes what the packed
file computes.

Schemes, by name:
    f16        every value rounded to float16
    i8c        symmetric int8 per output channel (codes -127..127), absmax scale
    s{b}c      symmetric b-bit per output channel, MSE-searched clip (codes -(2^(b-1)-1)..)
    s{b}g{G}   the same per group of G along K, fp16 scales
    a{b}g{G}   asymmetric b-bit per group of G (MatMulNBits-style: (code - zp) * scale,
               zp a b-bit code), fp16 scale
Scales are fp16 (rounded) in every scheme, so the fake values match a file
that stores fp16 scales.
"""
from __future__ import annotations

import re

import numpy as np

CLIP_GRID = np.linspace(1.0, 0.5, 26)


def _f16(x):
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def _sym(w: np.ndarray, bits: int, search: bool):
    """w: [R, n] rows quantized independently. Returns (codes int, scales [R])."""
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(w).max(1)
    amax[amax == 0] = 1e-8
    best_s = _f16(amax / qmax)
    best_s[best_s == 0] = 6e-8
    if search:
        best_err = np.full(len(w), np.inf)
        for f in CLIP_GRID:
            s = _f16(amax * f / qmax)
            s[s == 0] = 6e-8
            q = np.clip(np.round(w / s[:, None]), -qmax, qmax)
            err = ((q * s[:, None] - w) ** 2).sum(1)
            better = err < best_err
            best_err[better], best_s[better] = err[better], s[better]
    q = np.clip(np.round(w / best_s[:, None]), -qmax, qmax).astype(np.int32)
    return q, best_s


def _asym(w: np.ndarray, bits: int, search: bool):
    """Rows of w: codes 0..2^b-1, value (code - zp) * scale."""
    L = 2 ** bits - 1
    lo, hi = np.minimum(w.min(1), 0), np.maximum(w.max(1), 0)
    best = None
    grid = CLIP_GRID if search else [1.0]
    for f in grid:
        s = _f16((hi - lo) * f / L)
        s[s == 0] = 6e-8
        zp = np.clip(np.round(-lo * f / s), 0, L)
        q = np.clip(np.round(w / s[:, None]) + zp[:, None], 0, L)
        err = (((q - zp[:, None]) * s[:, None] - w) ** 2).sum(1)
        if best is None:
            best = [err, s, zp]
        else:
            m = err < best[0]
            best[0][m], best[1][m], best[2][m] = err[m], s[m], zp[m]
    _, s, zp = best
    q = np.clip(np.round(w / s[:, None]) + zp[:, None], 0, L).astype(np.int32)
    return q, s, zp.astype(np.int32)


def quantize(w2: np.ndarray, scheme: str) -> dict:
    """w2 [O, K] float32 -> {"fake": [O, K], "bits": total storage bits estimate, ...}."""
    O, K = w2.shape
    if scheme == "f32":
        return {"fake": w2, "bits": 32 * w2.size}
    if scheme == "f16":
        return {"fake": _f16(w2), "bits": 16 * w2.size}
    if scheme == "i8c":
        q, s = _sym(w2, 8, False)
        return {"fake": q * s[:, None], "codes": q, "scales": s, "bits": 8 * w2.size + 16 * O}
    m = re.fullmatch(r"([sa])(\d)(c|g(\d+))", scheme)
    if not m:
        raise ValueError(scheme)
    kind, bits, G = m.group(1), int(m.group(2)), int(m.group(4) or 0)
    if G == 0 or G >= K:
        rows, Gk, ng = w2, K, 1
    else:
        ng = -(-K // G)
        pad = np.zeros((O, ng * G), np.float32)
        pad[:, :K] = w2
        rows, Gk = pad.reshape(O * ng, G), G
    if kind == "s":
        q, s = _sym(rows, bits, True)
        fake = q * s[:, None]
        extra = 16
    else:
        q, s, zp = _asym(rows, bits, True)
        fake = (q - zp[:, None]) * s[:, None]
        extra = 16 + bits
    fake = fake.reshape(O, ng * Gk)[:, :K]
    return {"fake": fake.astype(np.float32), "codes": q, "scales": s,
            "bits": bits * w2.size + extra * len(rows)}


# ---------------------------------------------------------------- ONNX arms


def weight_axes(model) -> dict:
    """initializer name -> how to view it as [O, K]: ("conv"), ("matmul") or ("gemm", transB)."""
    out = {}
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1:
            out.setdefault(n.input[1], ("conv",))
        elif n.op_type == "MatMul":
            out.setdefault(n.input[1], ("matmul",))
        elif n.op_type == "Gemm":
            tb = next((a.i for a in n.attribute if a.name == "transB"), 0)
            out.setdefault(n.input[1], ("gemm", tb))
    return out


def to_ok(a: np.ndarray, how) -> np.ndarray:
    if how[0] == "conv" or (how[0] == "gemm" and how[1]):
        return a.reshape(a.shape[0], -1)
    return a.reshape(-1, a.shape[-1]).T  # MatMul [K, N] / Gemm transB=0


def from_ok(w2: np.ndarray, how, shape) -> np.ndarray:
    if how[0] == "conv" or (how[0] == "gemm" and how[1]):
        return w2.reshape(shape)
    return w2.T.reshape(shape)


def fake_model(src: str, dst: str, scheme, min_numel: int = 1024, other: str = "f16") -> dict:
    """Copy an ONNX file with every weight (Conv/MatMul/Gemm initializer of at least
    min_numel values) fake-quantized by `scheme` (a name, or a callable name -> scheme),
    other float initializers by `other`. Returns {"bits": total, "tensors": {...}}."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(src)
    axes = weight_axes(m)
    total, info = 0, {}
    for t in m.graph.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype != np.float32:
            continue
        how = axes.get(t.name)
        sch = scheme(t.name) if callable(scheme) else scheme
        if how is not None and a.size >= min_numel and a.ndim >= 2:
            r = quantize(to_ok(a, how), sch)
            new = from_ok(r["fake"], how, a.shape)
            info[t.name] = {"scheme": sch, "shape": list(a.shape), "bits": r["bits"],
                            "rel_err": float(np.linalg.norm(new - a) / max(np.linalg.norm(a), 1e-12))}
        else:
            r = quantize(a.reshape(1, -1), other)
            new = r["fake"].reshape(a.shape)
        total += r["bits"]
        t.CopyFrom(numpy_helper.from_array(new.astype(np.float32), t.name))
    onnx.save(m, dst)
    return {"bits": total, "bytes": total / 8, "tensors": info}
