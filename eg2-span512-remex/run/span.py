"""Step 1-2: do EG2 q4 ONNX outputs lie in a 512-dim span, and which basis spans it?

Usage: python3 -I run/span.py

model_q4.onnx ends in MatMulNBits /model/embedding_projection (K=512 -> N=768, 4-bit, block 32,
no bias), then masked mean pooling and an L2 norm. Both are linear or a per-row scale, so every
output should lie in the column space of the dequantized 768 x 512 projection whatever the
quantization did to the weights. Two bases, neither fitted on the evaluated vectors:
  proj     left singular vectors of the dequantized projection weight (data-free)
  heldout  top 512 right singular vectors (uncentred) of the 30,000 msmarco_train query
           targets: non-tech MS MARCO queries, a different domain and prompt from the posts
Checks: singular values of the held-out set and of the posts (rank 512 shows as a cliff after
index 511), the norm of each vector outside each basis for posts and both test query sets, and
the principal angles between the two bases. Writes out/basis_proj.npy, out/basis_heldout.npy
(512 x 768, rows orthonormal) and results/span.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
HF = HERE.parent / "eg2-query-student" / "out" / "hf"
NAME = "model_embedding_projection_MatMul_weight"


def nibbles(b):
    """uint8 [..., m] -> uint8 [..., 2m], low nibble first (MatMulNBits order)."""
    return np.stack([b & 0x0F, b >> 4], -1).reshape(*b.shape[:-1], -1)


def projection_weight():
    m = onnx.load(str(HF / "model_q4.onnx"), load_external_data=True)
    init = {i.name: i for i in m.graph.initializer}
    node = next(n for n in m.graph.node if n.name == "/model/embedding_projection/MatMul_Quant")
    attr = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    K, N, bs = attr["K"], attr["N"], attr["block_size"]
    q = numpy_helper.to_array(init[NAME + "_quant"])           # N x blocks x bs/2
    sc = numpy_helper.to_array(init[NAME + "_scales"]).astype(np.float64)   # N x blocks
    zp = nibbles(numpy_helper.to_array(init[NAME + "_zp"]))[:, : q.shape[1]]  # N x blocks
    vals = nibbles(q).astype(np.float64)                       # N x blocks x bs
    W = (vals - zp[..., None]) * sc[..., None]
    return W.reshape(N, -1)[:, :K], attr                       # y = h @ W.T, W is N x K


def outside(X, B):
    """Norm of each row of X outside the row space of B (orthonormal rows)."""
    return np.linalg.norm(X - (X @ B.T) @ B, axis=1)


def stats(r):
    return {"max": float(r.max()), "median": float(np.median(r))}


def main() -> int:
    W, attr = projection_weight()
    U, sw, _ = np.linalg.svd(W, full_matrices=False)
    Bp = U.T                                                    # 512 x 768
    H = np.load(OUT / "qt" / "msmarco_train.eg2.npy").astype(np.float64)
    _, sh, Vt = np.linalg.svd(H, full_matrices=False)
    Bh = Vt[:512]
    X = np.load(OUT / "docs.npy").astype(np.float64)
    live = np.linalg.norm(X, axis=1) > 0
    X = X[live]
    _, sx, _ = np.linalg.svd(X, full_matrices=False)
    tests = {k: np.load(OUT / "qt" / f"{k}.eg2.npy").astype(np.float64) for k in ("real_test", "synth_test")}
    cos = np.linalg.svd(Bp @ Bh.T, compute_uv=False)            # cosines of principal angles
    res = {
        "projection": {"node": "/model/embedding_projection/MatMul_Quant", **attr,
                       "weight_shape": list(W.shape), "rank_sv_ratio_last": float(sw[-1] / sw[0])},
        "heldout": {"set": "msmarco_train", "n": len(H),
                    "sv_ratio_512": float(sh[511] / sh[0]), "sv_ratio_513": float(sh[512] / sh[0])},
        "posts": {"n": int(live.sum()), "sv_ratio_512": float(sx[511] / sx[0]),
                  "sv_ratio_513": float(sx[512] / sx[0])},
        "principal_angle_cos_min": float(cos.min()),
        "outside_proj": {"posts": stats(outside(X, Bp)), **{k: stats(outside(v, Bp)) for k, v in tests.items()},
                         "heldout": stats(outside(H, Bp))},
        "outside_heldout": {"posts": stats(outside(X, Bh)), **{k: stats(outside(v, Bh)) for k, v in tests.items()}},
        "prefix512_energy_posts": float(np.mean(np.sum(X[:, :512] ** 2, 1))),
    }
    np.save(OUT / "basis_proj.npy", Bp.astype(np.float32))
    np.save(OUT / "basis_heldout.npy", Bh.astype(np.float32))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "span.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
