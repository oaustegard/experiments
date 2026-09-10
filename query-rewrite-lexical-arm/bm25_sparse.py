"""Streaming BM25 (Lucene variant) over a corpus too large for bm25s in RAM.

bm25s holds every token id as a Python list of lists — about 8 GB for this
corpus — and the chunked arm adds the chunk strings on top of that. This builds
the same index in two streaming passes into a scipy CSC matrix, which is ~1 GB
for the same corpus.

`validate()` reproduces bm25s rankings on a small corpus so the reimplementation
is checked against the library rather than trusted.
"""
import re
import numpy as np
import scipy.sparse as sp
from bm25s.stopwords import STOPWORDS_EN

K1, B = 1.5, 0.75
_PAT = re.compile(r"(?u)\b\w\w+\b")
_STOP = frozenset(STOPWORDS_EN)


class Analyzer:
    """Matches bm25s.tokenize(lower=True, stopwords='en', stemmer=snowball)."""

    def __init__(self, stemmer=None, stopwords=True):
        self.stemmer = stemmer
        self.stop = _STOP if stopwords else frozenset()
        self._cache = {}

    def __call__(self, text):
        out, cache, stem, stop = [], self._cache, self.stemmer, self.stop
        for w in _PAT.findall(text.lower()):
            if w in stop:
                continue
            if stem is None:
                out.append(w)
                continue
            s = cache.get(w)
            if s is None:
                s = cache[w] = stem.stemWord(w)
            out.append(s)
        return out


def build(stream_fn, n_units, analyzer, log=print):
    """Two passes over stream_fn() -> CSC of BM25 scores, shape (n_units, vocab)."""
    log("pass 1: vocabulary, document frequency, lengths")
    vocab, df, dl = {}, [], np.zeros(n_units, dtype=np.int32)
    nnz = 0
    for i, text in enumerate(stream_fn()):
        toks = analyzer(text)
        dl[i] = len(toks)
        seen = set()
        for t in toks:
            j = vocab.get(t)
            if j is None:
                j = vocab[t] = len(vocab)
                df.append(0)
            if j not in seen:
                seen.add(j)
                df[j] += 1
        nnz += len(seen)
        if i % 100000 == 0:
            log(f"  p1 {i}/{n_units} vocab={len(vocab)} nnz={nnz}")
    df = np.asarray(df, dtype=np.int64)
    V, N = len(vocab), n_units
    l_avg = float(dl.mean())
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    log(f"  vocab={V} nnz={nnz} avg_len={l_avg:.1f}")

    log("pass 2: scored triplets")
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    k = 0
    for i, text in enumerate(stream_fn()):
        tf = {}
        for t in analyzer(text):
            j = vocab[t]
            tf[j] = tf.get(j, 0) + 1
        if tf:
            m = len(tf)
            js = np.fromiter(tf.keys(), dtype=np.int32, count=m)
            fs = np.fromiter(tf.values(), dtype=np.float32, count=m)
            denom = K1 * ((1 - B) + B * dl[i] / l_avg)
            rows[k:k + m] = i
            cols[k:k + m] = js
            data[k:k + m] = idf[js] * (fs / (denom + fs))
            k += m
        if i % 100000 == 0:
            log(f"  p2 {i}/{n_units}")
    mat = sp.csc_matrix((data[:k], (rows[:k], cols[:k])), shape=(N, V), dtype=np.float32)
    del rows, cols, data
    log(f"  csc built: {mat.nnz} nnz, {mat.data.nbytes/2**30:.2f} GiB data")
    return mat, vocab


def retrieve(mat, vocab, queries, analyzer, k=10):
    """Top-k unit indices and scores per query. Sums the query terms' columns."""
    N = mat.shape[0]
    idx = np.zeros((len(queries), k), dtype=np.int64)
    sc = np.zeros((len(queries), k), dtype=np.float32)
    for qi, q in enumerate(queries):
        acc = np.zeros(N, dtype=np.float32)
        for t in analyzer(q):
            j = vocab.get(t)
            if j is None:
                continue
            s, e = mat.indptr[j], mat.indptr[j + 1]
            np.add.at(acc, mat.indices[s:e], mat.data[s:e])
        top = np.argpartition(-acc, min(k, N - 1))[:k]
        top = top[np.argsort(-acc[top])]
        idx[qi], sc[qi] = top, acc[top]
    return idx, sc


def validate(texts, queries, stemmer, k=10):
    """Assert this implementation ranks identically to bm25s on a small corpus."""
    import bm25s
    an = Analyzer(stemmer)
    mat, vocab = build(lambda: iter(texts), len(texts), an, log=lambda *a: None)
    mine_i, mine_s = retrieve(mat, vocab, queries, an, k=k)

    tok = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    r = bm25s.BM25(method="lucene", k1=K1, b=B)
    r.index(tok, show_progress=False)
    qtok = bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False)
    ref_i, ref_s = r.retrieve(qtok, k=k, show_progress=False)

    same_rank = int((np.asarray(ref_i) == mine_i).all())
    max_abs = float(np.abs(np.asarray(ref_s) - mine_s).max())
    return dict(identical_ranking=bool(same_rank), max_score_diff=max_abs)
