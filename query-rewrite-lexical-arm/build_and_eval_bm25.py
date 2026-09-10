"""Arms A1/A2: BM25 over EnterpriseRAG-Bench, chunked and whole-document.

Zero LLM calls, zero embeddings. Metrics are computed against the benchmark's
own expected_doc_ids, so no judge model is involved.

One arm per process. bm25s holds every token id as a Python list of lists,
which is ~8 GB for this corpus, so running both arms in one process exhausts
the container's 15 GB. Usage:

    python3 build_and_eval_bm25.py --arm A2
    python3 build_and_eval_bm25.py --arm A1

ERB_LIMIT=n subsets the corpus for a smoke test and labels the output as one.
"""
import argparse, gzip, hashlib, json, os, time
import numpy as np
import pyarrow.parquet as pq
import bm25s

DATA = "/home/user/erb-data"
WORK = "/home/user/erb-work"
OUT = os.path.dirname(os.path.abspath(__file__))
CHUNK_WORDS, CHUNK_STRIDE = 380, 340   # ~512 tokens / 50 overlap
TOPK_DOCS = 10
CHUNK_POOL = 200                        # chunks retrieved before doc collapse

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(WORK, exist_ok=True)

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

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


def aggregate(rows):
    ks = ["hit10", "hit1", "recall10", "mrr10", "ndcg10"]
    return {k: round(100 * float(np.mean([r[k] for r in rows])), 2) for k in ks}


def load_queries():
    d = pq.read_table(f"{DATA}/questions_test.parquet").to_pydict()
    qs = [dict(qid=q, qtype=t, text=x, gold=list(g))
          for q, t, x, g in zip(d["question_id"], d["question_type"],
                                d["question"], d["expected_doc_ids"])
          if g is not None and len(g) > 0]
    log(f"queries with qrels: {len(qs)} of {len(d['question_id'])}")
    return qs


def load_corpus(queries, limit):
    """Returns doc_ids and, in `texts`, title+content joined in place."""
    log("reading documents parquet ...")
    t = pq.read_table(f"{DATA}/documents_test.parquet")
    doc_ids = t.column("doc_id").to_pylist()
    titles = t.column("title").to_pylist()
    texts = t.column("content").to_pylist()
    del t
    if limit:
        keep = {g for q in queries for g in q["gold"]}
        gold_ix = [i for i, d in enumerate(doc_ids) if d in keep]
        filler = [i for i in range(len(doc_ids)) if doc_ids[i] not in keep][:limit]
        ix = sorted(set(gold_ix) | set(filler))
        doc_ids = [doc_ids[i] for i in ix]
        titles = [titles[i] for i in ix]
        texts = [texts[i] for i in ix]
        log(f"ERB_LIMIT: subset to {len(doc_ids)} docs — SMOKE TEST, not a result")
    for i in range(len(texts)):                    # in place: no second list
        texts[i] = f"{titles[i] or ''}\n{texts[i] or ''}"
    del titles
    log(f"documents: {len(doc_ids)}")
    return doc_ids, texts


def census(doc_ids, texts, queries):
    lens = np.fromiter((len(t) for t in texts), dtype=np.int64, count=len(texts))
    seen, dup = set(), 0
    for t in texts:
        h = hashlib.blake2b(t.encode(), digest_size=16).digest()
        if h in seen:
            dup += 1
        else:
            seen.add(h)
    have = set(doc_ids)
    absent = len({g for q in queries for g in q["gold"]} - have)
    log(f"chars: mean {lens.mean():.0f} median {np.median(lens):.0f} max {lens.max()}")
    log(f"exact duplicate docs: {dup} ({dup/len(texts):.2%}) | gold ids absent: {absent}")
    return dict(exact_dup_docs=dup, gold_ids_absent=absent,
                mean_doc_chars=int(lens.mean()))


def chunk(text):
    w = text.split()
    if len(w) <= CHUNK_WORDS:
        return [text] if w else [""]
    return [" ".join(w[s:s + CHUNK_WORDS])
            for s in range(0, max(1, len(w) - CHUNK_STRIDE + 1), CHUNK_STRIDE)]


def run(arm, doc_ids, texts, queries):
    if arm == "A1":
        log("A1: chunking (frees document text as it goes) ...")
        units, owner = [], []
        for i in range(len(texts)):
            for c in chunk(texts[i]):
                units.append(c); owner.append(i)
            texts[i] = ""                          # release as we go
            if i % 100000 == 0:
                log(f"  chunked {i}/{len(doc_ids)} -> {len(units)}")
        owner = np.asarray(owner, dtype=np.int32)
        log(f"A1: {len(units)} chunks ({len(units)/len(doc_ids):.2f} per doc)")
        k = CHUNK_POOL
    else:
        units, owner, k = texts, None, TOPK_DOCS

    log(f"{arm}: tokenizing {len(units)} units ...")
    tok = bm25s.tokenize(units, stopwords="en", stemmer=STEMMER, show_progress=False)
    if arm == "A1":
        units.clear()
    log(f"{arm}: indexing ...")
    r = bm25s.BM25(method="lucene")
    r.index(tok, show_progress=False)
    del tok

    log(f"{arm}: retrieving ...")
    qtok = bm25s.tokenize([q["text"] for q in queries], stopwords="en",
                          stemmer=STEMMER, show_progress=False)
    idx, sc = r.retrieve(qtok, k=min(k, len(doc_ids)), show_progress=False)

    rows = []
    for i, q in enumerate(queries):
        if arm == "A1":
            best = {}
            for j, s in zip(idx[i], sc[i]):
                d = int(owner[int(j)])
                if s > best.get(d, -1e9):
                    best[d] = float(s)
            order = sorted(best, key=lambda d: -best[d])[:TOPK_DOCS]
            ranked, scores = [doc_ids[d] for d in order], [best[d] for d in order]
        else:
            ranked = [doc_ids[int(j)] for j in idx[i]]
            scores = [float(s) for s in sc[i]]
        rows.append(dict(qid=q["qid"], qtype=q["qtype"], ranked=ranked,
                         scores=scores, gold=q["gold"], **metrics(ranked, q["gold"])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A1", "A2"], required=True)
    a = ap.parse_args()
    limit = int(os.environ.get("ERB_LIMIT", "0"))
    name = {"A1": "A1_chunked", "A2": "A2_whole_doc"}[a.arm]

    queries = load_queries()
    doc_ids, texts = load_corpus(queries, limit)
    cen = census(doc_ids, texts, queries)

    t0 = time.time()
    rows = run(a.arm, doc_ids, texts, queries)
    secs = round(time.time() - t0, 1)

    by_type = {t: dict(n=sum(1 for r in rows if r["qtype"] == t),
                       **aggregate([r for r in rows if r["qtype"] == t]))
               for t in sorted({r["qtype"] for r in rows})}
    res = dict(arm=name, smoke_test=bool(limit),
               config=dict(stemmer=STEMMER_NAME, method="lucene", stopwords="en",
                           chunk_words=CHUNK_WORDS, chunk_stride=CHUNK_STRIDE,
                           chunk_pool=CHUNK_POOL, topk_docs=TOPK_DOCS,
                           n_docs=len(doc_ids), n_queries=len(queries),
                           bm25s=bm25s.__version__, **cen),
               overall=aggregate(rows), by_type=by_type, seconds=secs)

    suffix = ".smoke" if limit else ""
    json.dump(res, open(f"{OUT}/results/{name}{suffix}.json", "w"), indent=2)
    with gzip.open(f"{OUT}/results/{name}{suffix}_runs.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"{name}: {json.dumps(res['overall'])}  ({secs}s)")
    log("DONE")
    open(f"{WORK}/{name}{suffix}.done", "w").write("ok")
