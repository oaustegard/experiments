"""float32 numpy model of the kernels in SPEC.md, in the kernels' order of
operations (rotate first, subtract R.mu after; pack into u32 words by the
SPEC's bit formula). Exists to check the folded math and the checker before
any WGSL is written.

Usage: python3 -I oracle/sim_tail.py FIXTURES OUTDIR
"""
import json
import os
import sys

import numpy as np

f32 = np.float32


def main():
    fx, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    p = json.load(open(os.path.join(fx, "params.json")))
    n, q, d, k = p["n_docs"], p["n_queries"], p["d"], p["k"]
    L = lambda f, dt, sh: np.fromfile(os.path.join(fx, f), dt).reshape(sh)
    R = L("R.f32", f32, (d, d))
    Rmu = L("Rmu.f32", f32, (d,))
    mu = L("mu.f32", f32, (d,))
    X = L("docs_in.f32", f32, (n, d))
    Q = L("queries_in.f32", f32, (q, d))
    b = np.array(p["boundaries"], f32)
    c = np.array(p["centroids"], f32)
    kappa = f32(p["kappa"])

    x = X / np.maximum(np.sqrt((X * X).sum(1, dtype=f32)), f32(1e-8))[:, None]
    t = (x @ R.T).astype(f32) - Rmu
    ln = np.sqrt((t * t).sum(1, dtype=f32))
    md = (t * Rmu).sum(1, dtype=f32)
    u = t / np.maximum(ln, f32(1e-8))[:, None]
    codes = ((u > b[0]).astype(np.uint32) + (u > b[1]) + (u > b[2])).astype(np.uint32)
    v = c[codes]
    vlen = np.sqrt((v * v).sum(1, dtype=f32))
    scale = ln / (np.maximum(vlen, f32(1e-12)) * kappa)
    bias = md - scale * (v * Rmu).sum(1, dtype=f32)

    # pack: word w, code 16w + 4bb + m at shift 8bb + 6 - 2m
    W = d // 16
    words = np.zeros((n, W), np.uint32)
    for w in range(W):
        for bb in range(4):
            for m in range(4):
                words[:, w] |= codes[:, 16 * w + 4 * bb + m] << np.uint32(8 * bb + 6 - 2 * m)
    packed = words.astype("<u4").view(np.uint8).reshape(n, -1)

    qn = Q / np.maximum(np.sqrt((Q * Q).sum(1, dtype=f32)), f32(1e-8))[:, None]
    qr = (qn @ R.T).astype(f32)
    off = (qn * mu).sum(1, dtype=f32)
    lut = qr[:, :, None] * c[None, None, :]                  # [q, d, 4]
    acc = np.zeros((q, n), f32)
    for j in range(d):
        acc += lut[:, j, :][:, codes[:, j]]
    scores = scale[None] * acc + bias[None] + off[:, None]
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int32)

    hidden = L("hidden.f32", f32, (p["pool_b"], p["pool_s"], d))
    mask = L("mask.u32", np.uint32, (p["pool_b"], p["pool_s"]))
    pooled = (hidden * mask[..., None]).sum(1, dtype=f32) / mask.sum(1, keepdims=True).astype(f32)

    for name, arr in {"packed.u8": packed, "scale.f32": scale, "bias.f32": bias,
                      "len.f32": ln, "mdot.f32": md, "scores.f32": scores,
                      "topk.i32": top, "pooled.f32": pooled}.items():
        np.ascontiguousarray(arr).tofile(os.path.join(out, name))


if __name__ == "__main__":
    main()
