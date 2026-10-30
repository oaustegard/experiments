"""Build the 3-bit, 512-dim rotated-span index the search page loads by default.

Usage: python3 -I run/build_bsky_span512.py

Why: eg2-span512-remex found that EG2 vectors lie in a 512-dim span, and that 3-bit remex codes of
the 512 rotated coordinates (192 B per post, the size of the 2-bit 768 index) raise recall@10
against exact search from 0.837 to 0.894.

Input: ../eg2-span512-remex/out/docs.npy (fp32 768-d EG2 q4 vectors of the 61,686 posts, in
posts.json order, zero rows for the 12 deleted posts; that experiment's run/embed_docs.py) and
out/basis_proj.npy (512 x 768, from the dequantized projection weight; its run/span.py).

Steps: z = B x for every live post; remex Quantizer(512, bits=3, seed=42, mean=mean of z) codes,
scale and bias. Deleted posts get zero codes, scale 0 and bias -1e30, so they never rank.
Writes web/bsky/:
  span512.json      d, bits, centroids, mu768 (= B^T mu512, so the query offset q . mu512 is a
                    dot product with the 768-d query), file names, and parity: test queries with
                    remex search_adc top-10 and scores and the exact fp32 768 top-10
  codes512.u8.txt   n x 48 little-endian u32 words, doc-major, bit-plane layout: for block b of
                    32 dims, words 3b, 3b+1, 3b+2 hold bit 2, 1, 0 of the code of dim 32b+t at bit t
  aux512.f32.txt    n x 2 float32 (scale, bias)
  qmat512.f32.txt   M^T as 768 x 512 float32, M = R B (remex's rotation folded into the basis), so
                    the page computes (R B q)_j = sum_k M^T[k, j] q_k in one pass
index.json gains "span512": "span512.json". The 768-d 2-bit files stay, for ?span=0, ?build=1's
?compare=1, and eg2-query-student.
"""
import json
import sys
from pathlib import Path

import numpy as np
import remex

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "bsky"
SPAN = ROOT.parent / "eg2-span512-remex"
sys.path.insert(0, str(ROOT.parent / "eg2-query-student" / "run"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bsky_index import PARITY_QUERIES, QUERY  # noqa: E402
from common import Teacher  # noqa: E402

D, BITS = 512, 3


def bitplanes(idx):
    """(n, 512) codes in 0..7 -> (n, 48) uint32 words in the layout described above."""
    n = len(idx)
    blocks = idx.reshape(n, 16, 32).astype(np.uint32)
    weights = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    words = np.empty((n, 16, 3), np.uint32)
    for p, bit in enumerate((2, 1, 0)):
        words[:, :, p] = (((blocks >> bit) & 1) * weights).sum(2, dtype=np.uint64).astype(np.uint32)
    return words.reshape(n, 48)


def main() -> int:
    X = np.load(SPAN / "out" / "docs.npy")
    B = np.load(SPAN / "out" / "basis_proj.npy").astype(np.float64)
    meta = json.loads((WEB / "index.json").read_text())
    assert len(X) == meta["n"], (len(X), meta["n"])
    live = np.linalg.norm(X, axis=1) > 0
    Z = (X[live] @ B.T).astype(np.float32)
    mu = Z.mean(0).astype(np.float32)
    qz = remex.Quantizer(D, bits=BITS, seed=42, mean=mu)
    cv = qz.encode(Z)

    idx = np.zeros((len(X), D), np.uint8)
    idx[live] = cv.indices
    aux = np.zeros((len(X), 2), "<f4")
    aux[:, 1] = -1e30
    aux[live, 0] = qz._score_norms(cv)
    aux[live, 1] = qz._row_bias(cv)
    words = bitplanes(idx)
    # the layout round-trips
    back = np.zeros_like(idx)
    w = words.reshape(-1, 16, 3)
    for p, bit in enumerate((2, 1, 0)):
        back |= (((w[:, :, p, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8) << bit).reshape(-1, D)
    assert (back == idx).all()

    Rt = qz._rotate_rows(np.eye(D, dtype=np.float32)).astype(np.float64)   # row i = R e_i
    M = Rt.T @ B                                                          # 512 x 768
    mu768 = B.T @ mu.astype(np.float64)

    t = Teacher(threads=4)
    Q = t.embed(PARITY_QUERIES, prompt=QUERY)
    Xl = np.where(live[:, None], X, 0)
    parity = []
    for q, v in zip(PARITY_QUERIES, Q):
        ri, rs = qz.search_adc(cv, (B @ v).astype(np.float32), k=10)
        rows = np.flatnonzero(live)[ri]
        # the page's formula, in float64, against remex's own scores
        qr = M @ v
        mine = aux[rows, 0] * (qr[None, :] * qz.centroids[idx[rows]]).sum(1) + aux[rows, 1] + v @ mu768
        assert np.abs(mine - rs).max() < 1e-4, np.abs(mine - rs).max()
        parity.append({"query": q, "remex_top10": [int(i) for i in rows], "remex_scores": [float(s) for s in rs],
                       "exact_top10": [int(i) for i in np.argsort(-(Xl @ v))[:10]]})

    (WEB / "codes512.u8.txt").write_bytes(words.astype("<u4").tobytes())
    (WEB / "aux512.f32.txt").write_bytes(aux.tobytes())
    (WEB / "qmat512.f32.txt").write_bytes(M.T.astype("<f4").tobytes())
    (WEB / "span512.json").write_text(json.dumps({
        "n": int(len(X)), "live": int(live.sum()), "d": D, "bits": BITS, "seed": 42, "rotation": "rht",
        "remex_version": remex.__version__, "basis": "eg2-span512-remex/out/basis_proj.npy (q4 projection weight)",
        "layout": "bitplane: block b of 32 dims -> words 3b,3b+1,3b+2 = bit 2,1,0; dim 32b+t at bit t",
        "centroids": [float(c) for c in qz.centroids], "mu768": [float(v) for v in mu768],
        "files": {"codes": "codes512.u8.txt", "aux": "aux512.f32.txt", "qmat": "qmat512.f32.txt"},
        "parity": parity,
    }, separators=(",", ":")))
    meta["span512"] = "span512.json"
    (WEB / "index.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(json.dumps({"n": len(X), "live": int(live.sum()), "codes_bytes": words.nbytes,
                      "centroids": [round(float(c), 4) for c in qz.centroids]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
