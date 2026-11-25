"""Black-box tests of compress.py, written by gemini-4-hydrogen from a spec (DELEGATION.md job 1)."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compress as C


def _make_csr(n_rows, n_cols, min_nnz, max_nnz, seed=0, exp_scale=False):
    rng = np.random.default_rng(seed)
    indptr = [0]
    indices = []
    data = []
    for _ in range(n_rows):
        nnz = int(rng.integers(min_nnz, max_nnz + 1))
        cols = np.sort(rng.choice(n_cols, size=nnz, replace=False))
        if exp_scale:
            vals = np.exp(rng.normal(0.0, 0.7, size=nnz)).astype(np.float32)
        else:
            vals = rng.uniform(0.1, 10.0, size=nnz).astype(np.float32)
            vals += np.arange(nnz, dtype=np.float32) * 1e-3  # distinct values per row
        indices.append(cols)
        data.append(vals)
        indptr.append(indptr[-1] + nnz)
    return sp.csr_matrix(
        (np.concatenate(data) if data else np.array([], dtype=np.float32),
         np.concatenate(indices) if indices else np.array([], dtype=np.int32),
         np.array(indptr, dtype=np.int32)),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )


def test_topk_rows():
    D = _make_csr(50, 200, 0, 40, seed=42)
    for k in [1, 5, 16, None]:
        out = C.topk_rows(D, k)
        assert sp.isspmatrix_csr(out)
        assert out.shape == D.shape
        assert np.all(out.data > 0)
        for r in range(D.shape[0]):
            s0, e0 = D.indptr[r], D.indptr[r + 1]
            row_idx = D.indices[s0:e0]
            row_val = D.data[s0:e0]
            s1, e1 = out.indptr[r], out.indptr[r + 1]
            out_idx = out.indices[s1:e1]
            out_val = out.data[s1:e1]
            assert np.all(out_idx[:-1] < out_idx[1:])
            if k is None or len(row_idx) <= k:
                assert np.array_equal(out_idx, row_idx)
                assert np.array_equal(out_val, row_val)
            else:
                top_pos = np.argsort(row_val)[-k:]
                exp_idx = np.sort(row_idx[top_pos])
                exp_val = row_val[np.sort(top_pos)]
                assert np.array_equal(out_idx, exp_idx)
                assert np.array_equal(out_val, exp_val)


def test_varint_len():
    g = np.array([0, 127, 128, 16383, 16384, 50369])
    assert np.array_equal(C.varint_len(g), np.array([1, 1, 2, 2, 3, 3]))


def test_index_bytes():
    # row0: ids [5, 200, 50000] -> gaps [5, 195, 49800] -> varint lens [1, 2, 3] = 6
    # row1: ids [] -> 0 varint bytes
    # row2: ids [130] -> gaps [130] -> varint len [2] = 2
    rows = [0, 0, 0, 2]
    cols = [5, 200, 50000, 130]
    vals = [1.0, 2.0, 3.0, 4.0]
    Dk = sp.csr_matrix((vals, (rows, cols)), shape=(3, 60000), dtype=np.float32)
    # value_bits=4, scale_bytes=2:
    # row0: fixed = 6+2+2 = 10; varint = 6+2+2 = 10
    # row1: fixed = 2; varint = 2
    # row2: fixed = 2+1+2 = 5; varint = 2+1+2 = 5
    f_mean, v_mean, n_mean = C.index_bytes(Dk, value_bits=4, scale_bytes=2)
    assert np.isclose(f_mean, 17.0 / 3.0)
    assert np.isclose(v_mean, 17.0 / 3.0)
    assert np.isclose(n_mean, 4.0 / 3.0)
    # value_bits=0, scale_bytes=0: row0 6/6, row1 0/0, row2 2/2
    f0, v0, n0 = C.index_bytes(Dk, value_bits=0, scale_bytes=0)
    assert np.isclose(f0, 8.0 / 3.0)
    assert np.isclose(v0, 8.0 / 3.0)
    assert np.isclose(n0, 4.0 / 3.0)


def _check_codebook_invariants(cb, bits, v):
    assert len(cb.levels) <= 2 ** bits
    assert np.all(cb.levels[:-1] < cb.levels[1:])
    assert len(cb.bounds) == len(cb.levels) - 1
    assert np.all(cb.bounds[:-1] <= cb.bounds[1:])
    out = cb(v)
    assert out.dtype == np.float32
    expected = cb.levels[np.searchsorted(cb.bounds, v)]
    assert np.allclose(out, expected)
    assert np.all(np.isin(out, cb.levels.astype(np.float32)))
    return out


def test_codebook_uniform():
    cb = C.Codebook("uniform", 2, np.array([0.1, 4.0], dtype=np.float32))
    assert np.allclose(cb.levels, [0.5, 1.5, 2.5, 3.5])
    v = np.array([0.2, 1.0, 2.01, 3.99, 10.0], dtype=np.float32)
    out = _check_codebook_invariants(cb, 2, v)
    assert np.allclose(out, [0.5, 0.5, 2.5, 3.5, 3.5])


def test_codebook_lloyd():
    rng = np.random.default_rng(7)
    sample = np.concatenate([
        rng.normal(1.0, 0.01, 1000),
        rng.normal(5.0, 0.01, 1000),
        rng.normal(9.0, 0.01, 1000),
    ]).astype(np.float32)
    cb = C.Codebook("lloyd", 2, sample)
    q = _check_codebook_invariants(cb, 2, sample)
    for center in [1.0, 5.0, 9.0]:
        assert np.min(np.abs(cb.levels - center)) < 0.05
    assert np.mean((sample - q) ** 2) < 0.01
    cells = np.searchsorted(cb.bounds, sample)
    for i, lvl in enumerate(cb.levels):
        pts = sample[cells == i]
        if len(pts) > 0:
            assert abs(lvl - np.mean(pts.astype(np.float64))) < 1e-6


def test_codebook_loglloyd():
    rng = np.random.default_rng(0)
    sample = np.exp(rng.uniform(-3.0, 1.0, 20000)).astype(np.float32)
    cb = C.Codebook("loglloyd", 3, sample)
    _check_codebook_invariants(cb, 3, sample[:100])
    assert np.all(cb.levels > 0)
    log_gaps = np.diff(np.log(cb.levels))
    assert np.max(log_gaps) / np.min(log_gaps) < 1.5


def test_quantize_docs():
    Dk = _make_csr(30, 500, 5, 50, seed=11, exp_scale=True)
    sample_rows = np.arange(30)
    for scale in ["none", "docmax", "l2"]:
        dec, cb = C.quantize_docs(Dk, "uniform", 4, scale, sample_rows)
        assert np.array_equal(dec.indices, Dk.indices)
        assert np.array_equal(dec.indptr, Dk.indptr)
        if scale == "docmax":
            assert np.max(cb.levels) <= 1.0 + 1e-6
            for r in range(Dk.shape[0]):
                s, e = Dk.indptr[r], Dk.indptr[r + 1]
                rmax = float(np.float16(np.max(Dk.data[s:e])))
                exp_row = rmax * cb(Dk.data[s:e] / rmax)
                assert np.allclose(dec.data[s:e], exp_row, rtol=1e-5, atol=1e-5)
        elif scale == "l2":
            for r in range(Dk.shape[0]):
                s, e = Dk.indptr[r], Dk.indptr[r + 1]
                orig_l2 = float(np.float16(np.linalg.norm(Dk.data[s:e])))
                dec_l2 = float(np.linalg.norm(dec.data[s:e]))
                assert abs(dec_l2 - orig_l2) / orig_l2 < 1e-3
    dec8, _ = C.quantize_docs(Dk, "uniform", 8, "docmax", sample_rows)
    for r in range(Dk.shape[0]):
        s, e = Dk.indptr[r], Dk.indptr[r + 1]
        rmax = float(np.max(Dk.data[s:e]))
        max_err = float(np.max(np.abs(dec8.data[s:e] - Dk.data[s:e])))
        assert max_err <= rmax / 256.0 + 1e-3 * rmax


def test_binary_docs():
    Dk = _make_csr(10, 50, 3, 15, seed=5, exp_scale=True)
    for scale in ["none", "docmax", "l2"]:
        out = C.binary_docs(Dk, scale)
        assert np.array_equal(out.indices, Dk.indices)
        assert np.array_equal(out.indptr, Dk.indptr)
        for r in range(Dk.shape[0]):
            s, e = Dk.indptr[r], Dk.indptr[r + 1]
            nnz = e - s
            if scale == "none":
                assert np.allclose(out.data[s:e], 1.0)
            elif scale == "docmax":
                rmax = float(np.float16(np.max(Dk.data[s:e])))
                assert np.allclose(out.data[s:e], rmax)
            elif scale == "l2":
                rl2 = float(np.float16(np.linalg.norm(Dk.data[s:e])))
                assert np.allclose(out.data[s:e], rl2 / np.sqrt(nnz))


def test_anserini_8bit():
    rows = [0, 0, 0, 1, 1]
    cols = [1, 3, 7, 2, 8]
    vals = [10.0, 0.01, 5.0, 0.019, 7.0]
    Dk = sp.csr_matrix((vals, (rows, cols)), shape=(2, 10), dtype=np.float32)
    out = C.anserini_8bit(Dk)
    hi = 10.0
    assert np.array_equal(out.indptr, [0, 2, 3])
    assert np.array_equal(out.indices, [1, 7, 8])
    exp_vals = np.array([round(10.0 / hi * 255) * hi / 255, round(5.0 / hi * 255) * hi / 255,
                         round(7.0 / hi * 255) * hi / 255], dtype=np.float32)
    assert np.allclose(out.data, exp_vals)


def test_metrics():
    top = np.array([
        [7, 1, 3, 0, 2, 4, 5, 6, 8, 9, 10, 11],
        [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ])
    qrels_idx = [{3: 1, 7: 2}, {11: 1, 99: 1}]
    ideal10 = np.array([2.0 / np.log2(2) + 1.0 / np.log2(3), 1.0 / np.log2(2) + 1.0 / np.log2(3)])
    ndcg10, r100 = C.metrics(top, qrels_idx, ideal10)
    assert np.allclose(ndcg10, [2.5 / ideal10[0], 1.0 / ideal10[1]])
    assert np.allclose(r100, [1.0, 0.5])


def test_boot_ci():
    delta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo, hi = C.boot_ci(delta, n=1000, seed=0)
    assert lo <= np.mean(delta) <= hi
    c = 3.25
    lo_c, hi_c = C.boot_ci(np.full(10, c), n=1000, seed=0)
    assert lo_c == c and hi_c == c


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("pass", name)
    print("ALL PASS")
