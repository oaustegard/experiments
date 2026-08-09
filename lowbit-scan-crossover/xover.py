"""Decompose the 'BLAS beats sign-bit Hamming below ~150k rows' crossover.

Question under test: is the crossover a property of the *tradeoff* (32x less data
but more work per byte) or of the *kernel implementation* (numpy expression graph
vs OpenBLAS sgemv)?
"""
import os, sys, time
import numpy as np

K = 256
W = K // 64          # 4 uint64 words per vector
REPS = 7
TOPK = 1000

rng = np.random.default_rng(0)


def make(n):
    X = rng.standard_normal((n, K), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    bits = (X > 0)
    C = np.packbits(bits, axis=1, bitorder='little').view(np.uint64)
    assert C.shape == (n, W), C.shape
    q = X[0].copy()
    qc = C[0].copy()
    return X, C, q, qc


def timeit(fn, reps=REPS):
    fn()                                    # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3, float(np.median(ts)) * 1e3


# ---------------- kernels ----------------

def f32_scan(X, q):
    return X @ q

def ham_naive(C, qc):
    """The kernel as written in the original note: 3 whole-array passes, 2 temporaries."""
    return np.bitwise_count(C ^ qc).sum(axis=1)

def ham_chunked(C, qc, chunk=8192, out=None, tmp=None):
    """Same arithmetic, blocked so intermediates stay in L1/L2, buffers reused."""
    n = C.shape[0]
    if out is None:
        out = np.empty(n, dtype=np.uint64)
    if tmp is None:
        tmp = np.empty((chunk, W), dtype=np.uint64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        blk = tmp[:j - i]
        np.bitwise_xor(C[i:j], qc, out=blk)
        np.bitwise_count(blk, out=blk)
        np.add.reduce(blk, axis=1, out=out[i:j])
    return out

def ham_matmul(Cb_f32, qb_f32):
    """Sign-bit Hamming expressed as a dot product on +-1 float32 -> BLAS does it.
    Storage stays 32 B/vec; this is the *decode-to-BLAS* variant for reference."""
    return Cb_f32 @ qb_f32


def topk(scores, k=TOPK, largest=True):
    s = scores if largest else -scores.astype(np.int32)
    return np.argpartition(s, -k)[-k:]


# ---------------- run ----------------

def bench(n):
    X, C, q, qc = make(n)
    r = {'n': n,
         'f32_GB': X.nbytes / 1e9,
         'ham_GB': C.nbytes / 1e9}

    r['f32_scan'] = timeit(lambda: f32_scan(X, q))[0]
    r['ham_naive'] = timeit(lambda: ham_naive(C, qc))[0]

    out = np.empty(n, dtype=np.uint64)
    tmp = np.empty((8192, W), dtype=np.uint64)
    r['ham_chunk'] = timeit(lambda: ham_chunked(C, qc, 8192, out, tmp))[0]

    # end-to-end incl. top-k selection (what the original note timed)
    sf = f32_scan(X, q)
    r['topk_f32'] = timeit(lambda: topk(sf))[0]
    sh = ham_naive(C, qc)
    r['topk_ham'] = timeit(lambda: topk(sh, largest=False))[0]

    r['e2e_f32'] = r['f32_scan'] + r['topk_f32']
    r['e2e_naive'] = r['ham_naive'] + r['topk_ham']
    r['e2e_chunk'] = r['ham_chunk'] + r['topk_ham']

    # correctness: chunked == naive
    assert np.array_equal(ham_chunked(C, qc).astype(np.int64), ham_naive(C, qc).astype(np.int64))
    del X, C
    return r


if __name__ == '__main__':
    ns = [int(x) for x in sys.argv[1:]] or [42_500, 250_000, 1_000_000]
    print(f"threads env: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}  numpy {np.__version__}")
    hdr = ('n', 'f32 GB', 'ham GB', 'f32scan', 'hamNaive', 'hamChunk',
           'topk_f32', 'topk_ham', 'e2e_f32', 'e2e_naive', 'e2e_chunk', 'x_naive', 'x_chunk')
    print(' '.join(f'{h:>10}' for h in hdr))
    for n in ns:
        r = bench(n)
        row = [f"{r['n']:>10,}", f"{r['f32_GB']:>10.3f}", f"{r['ham_GB']:>10.3f}"] + [
            f"{r[k]:>10.2f}" for k in ('f32_scan', 'ham_naive', 'ham_chunk',
                                       'topk_f32', 'topk_ham',
                                       'e2e_f32', 'e2e_naive', 'e2e_chunk')]
        row += [f"{r['e2e_f32']/r['e2e_naive']:>10.2f}", f"{r['e2e_f32']/r['e2e_chunk']:>10.2f}"]
        print(' '.join(row))
