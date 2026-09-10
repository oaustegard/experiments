"""Arm A1 (BM25 over chunks) via the streaming sparse index.

Chunks are regenerated per pass rather than held, so peak memory is the document
text plus the CSC matrix instead of both plus every chunk string.
"""
import argparse, gzip, json, os, time
import numpy as np
import pyarrow.parquet as pq
import bm25_sparse as BS

DATA = "/home/user/erb-data"
WORK = "/home/user/erb-work"
OUT = os.path.dirname(os.path.abspath(__file__))
TOPK_DOCS, CHUNK_POOL = 10, 200

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

try:
    import Stemmer
    STEMMER, STEMMER_NAME = Stemmer.Stemmer("english"), "snowball-english"
except ImportError:
    STEMMER, STEMMER_NAME = None, "none"


def metrics(ranked, gold):
    gs = set(gold)
    hits = [1 if d in gs else 0 for d in ranked[:TOPK_DOCS]]
    mrr = next((1.0 / r for r, h in enumerate(hits, 1) if h), 0.0)
    dcg = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
    idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(gs), TOPK_DOCS) + 1))
    return dict(hit10=float(any(hits)), hit1=float(hits[:1] == [1]),
                recall10=sum(hits) / len(gs), mrr10=mrr,
                ndcg10=dcg / idcg if idcg else 0.0)


def agg(rows):
    ks = ["hit10", "hit1", "recall10", "mrr10", "ndcg10"]
    return {k: round(100 * float(np.mean([r[k] for r in rows])), 2) for k in ks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=380)
    ap.add_argument("--stride", type=int, default=340)
    ap.add_argument("--name", default="A1_chunked")
    ap.add_argument("--collapse", choices=["docs","chunks"], default="docs",
                    help="docs: pool 200 chunks then take 10 distinct documents. "
                         "chunks: take the top 10 chunks and keep whatever documents they land on")
    ap.add_argument("--queries-json", dest="queries_json", default="", help="qid->text overrides")
    ap.add_argument("--subcorpus", default="", help="json list of doc_ids to keep")
    a = ap.parse_args()
    W, S = a.words, a.stride

    q = pq.read_table(f"{DATA}/questions_test.parquet").to_pydict()
    queries = [dict(qid=i, qtype=t, text=x, gold=list(g))
               for i, t, x, g in zip(q["question_id"], q["question_type"],
                                     q["question"], q["expected_doc_ids"])
               if g is not None and len(g) > 0]
    if a.queries_json:
        ov = json.load(open(a.queries_json))
        n_ov = sum(1 for x in queries if x["qid"] in ov)
        for x in queries:
            x["text"] = ov.get(x["qid"], x["text"])
        log(f"query overrides applied to {n_ov}/{len(queries)}")
    log(f"queries: {len(queries)}")

    log("reading documents ...")
    t = pq.read_table(f"{DATA}/documents_test.parquet")
    doc_ids = t.column("doc_id").to_pylist()
    titles = t.column("title").to_pylist()
    texts = t.column("content").to_pylist()
    del t
    for i in range(len(texts)):
        texts[i] = f"{titles[i] or ''}\n{texts[i] or ''}"
    del titles
    if a.subcorpus:
        keep = set(json.load(open(a.subcorpus)))
        ix = [i for i, d in enumerate(doc_ids) if d in keep]
        doc_ids = [doc_ids[i] for i in ix]
        texts = [texts[i] for i in ix]
        log(f"subcorpus: {len(doc_ids)} docs from {a.subcorpus}")
    N = len(doc_ids)
    log(f"documents: {N}")

    def parts(text):
        w = text.split()
        if len(w) <= W:
            return [text] if w else [""]
        return [" ".join(w[s:s + W]) for s in range(0, max(1, len(w) - S + 1), S)]

    log("counting chunks ...")
    counts = np.fromiter((len(parts(t)) for t in texts), dtype=np.int32, count=N)
    n_chunks = int(counts.sum())
    owner = np.repeat(np.arange(N, dtype=np.int32), counts)
    del counts
    log(f"chunks: {n_chunks} ({n_chunks/N:.2f} per doc)")

    def stream():
        for t in texts:
            yield from parts(t)

    an = BS.Analyzer(STEMMER)
    t0 = time.time()
    mat, vocab = BS.build(stream, n_chunks, an, log=log)
    log(f"retrieving (collapse={a.collapse}) ...")
    k_units = CHUNK_POOL if a.collapse == "docs" else TOPK_DOCS
    idx, sc = BS.retrieve(mat, vocab, [x["text"] for x in queries], an, k=k_units)

    rows = []
    for i, qq in enumerate(queries):
        if a.collapse == "chunks":
            order, seen = [], set()
            for j in idx[i]:                      # top-10 chunks, in rank order
                d = int(owner[int(j)])
                if d not in seen:
                    seen.add(d); order.append(d)
            scores = [float(s) for s in sc[i][:len(order)]]
        else:
            best = {}
            for j, s in zip(idx[i], sc[i]):
                d = int(owner[int(j)])
                if s > best.get(d, -1e9):
                    best[d] = float(s)
            order = sorted(best, key=lambda d: -best[d])[:TOPK_DOCS]
            scores = [best[d] for d in order]
        ranked = [doc_ids[d] for d in order]
        rows.append(dict(qid=qq["qid"], qtype=qq["qtype"], ranked=ranked,
                         scores=scores, gold=qq["gold"],
                         **metrics(ranked, qq["gold"])))
    secs = round(time.time() - t0, 1)

    res = dict(arm=a.name, smoke_test=False,
               config=dict(impl="bm25_sparse (validated identical to bm25s)",
                           stemmer=STEMMER_NAME, k1=BS.K1, b=BS.B,
                           chunk_words=W, chunk_stride=S, chunk_pool=CHUNK_POOL,
                           topk_docs=TOPK_DOCS, n_docs=N, n_chunks=n_chunks,
                           n_queries=len(queries), vocab=len(vocab), nnz=int(mat.nnz),
                           subcorpus=os.path.basename(a.subcorpus) or None,
                           collapse=a.collapse),
               overall=agg(rows),
               by_type={t: dict(n=sum(1 for r in rows if r["qtype"] == t),
                                **agg([r for r in rows if r["qtype"] == t]))
                        for t in sorted({r["qtype"] for r in rows})},
               seconds=secs)
    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump(res, open(f"{OUT}/results/{a.name}.json", "w"), indent=2)
    with gzip.open(f"{OUT}/results/{a.name}_runs.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"{a.name}: {json.dumps(res['overall'])} ({secs}s)")
    log("DONE")
    open(f"{WORK}/{a.name}.done", "w").write("ok")
