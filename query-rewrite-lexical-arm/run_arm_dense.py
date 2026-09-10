"""Arm B: dense retrieval with all-MiniLM-L6-v2, mean pooling, L2 normalised.

bge-base-en-v1.5 is the paper's own retriever and was the first choice, but it
measures 3.7 chunks/s on this container's four cores, which is 83 hours for the
corpus. MiniLM-L6-v2 runs at 45 chunks/s and the paper uses it as its own
retriever-robustness check in Appendix J, so it is the encoder they themselves
fall back to. The consequence is recorded in RESULTS.md: the +/-3 point anchor
to their 39.22 BGE baseline is not claimed, and only the internal comparison
between modalities is.

Encoding checkpoints to a memmap so a crash resumes instead of restarting.
"""
import argparse, gzip, json, os, time
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATA, WORK = "/home/user/erb-data", "/home/user/erb-work"
MODEL = "/home/user/minilm"
OUT = os.path.dirname(os.path.abspath(__file__))
DIM, MAXLEN, TOPK_DOCS, CHUNK_POOL = 384, 256, 10, 200

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


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


class Encoder:
    def __init__(self, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(f"{MODEL}/model.onnx", so, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.s.get_inputs()}
        self.tk = Tokenizer.from_file(f"{MODEL}/tokenizer.json")
        self.tk.enable_truncation(max_length=MAXLEN)
        self.tk.enable_padding(pad_id=0, pad_token="[PAD]")

    def __call__(self, texts, batch=16, out=None):
        out = np.empty((len(texts), DIM), dtype=np.float16) if out is None else out
        for i in range(0, len(texts), batch):
            e = self.tk.encode_batch(texts[i:i + batch])
            ids = np.asarray([x.ids for x in e], dtype=np.int64)
            am = np.asarray([x.attention_mask for x in e], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": am}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.s.run(None, feed)[0]
            m = am[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)      # mean pooling
            out[i:i + len(e)] = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", type=int, default=180)
    ap.add_argument("--stride", type=int, default=160)
    ap.add_argument("--name", default="B_dense")
    ap.add_argument("--subcorpus", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    W, S = a.words, a.stride
    tag = a.name

    q = pq.read_table(f"{DATA}/questions_test.parquet").to_pydict()
    queries = [dict(qid=i, qtype=t, text=x, gold=list(g))
               for i, t, x, g in zip(q["question_id"], q["question_type"],
                                     q["question"], q["expected_doc_ids"])
               if g is not None and len(g) > 0]
    log(f"queries: {len(queries)}")

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
        log(f"subcorpus: {len(doc_ids)} docs")
    N = len(doc_ids)

    def parts(text):
        w = text.split()
        if len(w) <= W:
            return [text] if w else [""]
        return [" ".join(w[s:s + W]) for s in range(0, max(1, len(w) - S + 1), S)]

    log("chunking ...")
    chunks, owner = [], []
    for i in range(N):
        for c in parts(texts[i]):
            chunks.append(c); owner.append(i)
        texts[i] = ""
    owner = np.asarray(owner, dtype=np.int32)
    n = len(chunks)
    log(f"chunks: {n} ({n/N:.2f} per doc)")

    enc = Encoder(a.threads)
    vpath = f"{WORK}/{tag}.vecs.f16.npy"
    ppath = f"{WORK}/{tag}.progress"
    if os.path.exists(vpath) and os.path.exists(ppath):
        vecs = np.lib.format.open_memmap(vpath, mode="r+")
        start = int(open(ppath).read().strip())
        assert vecs.shape == (n, DIM), f"shape {vecs.shape} != {(n, DIM)}"
        log(f"resuming at {start}/{n}")
    else:
        vecs = np.lib.format.open_memmap(vpath, mode="w+", dtype=np.float16, shape=(n, DIM))
        start = 0

    STEP, t0 = 20000, time.time()
    for s in range(start, n, STEP):
        e = min(s + STEP, n)
        vecs[s:e] = enc(chunks[s:e], batch=a.batch)
        vecs.flush(); open(ppath, "w").write(str(e))
        rate = (e - start) / (time.time() - t0)
        log(f"encoded {e}/{n}  {rate:.1f}/s  eta {(n-e)/rate/3600:.2f} h")
    del chunks
    log("encoding done; retrieving")

    qv = enc([x["text"] for x in queries], batch=8).astype(np.float32)
    best_all = [dict() for _ in queries]
    BLK = 200000
    for s in range(0, n, BLK):
        e = min(s + BLK, n)
        sims = qv @ np.asarray(vecs[s:e], dtype=np.float32).T          # (Q, blk)
        kk = min(CHUNK_POOL, e - s)
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for qi in range(len(queries)):
            for j in top[qi]:
                d = int(owner[s + int(j)])
                v = float(sims[qi, j])
                if v > best_all[qi].get(d, -1e9):
                    best_all[qi][d] = v
        del sims

    rows = []
    for i, qq in enumerate(queries):
        order = sorted(best_all[i], key=lambda d: -best_all[i][d])[:TOPK_DOCS]
        ranked = [doc_ids[d] for d in order]
        rows.append(dict(qid=qq["qid"], qtype=qq["qtype"], ranked=ranked,
                         scores=[best_all[i][d] for d in order], gold=qq["gold"],
                         **metrics(ranked, qq["gold"])))

    res = dict(arm=tag, smoke_test=False,
               config=dict(encoder="all-MiniLM-L6-v2", pooling="mean", dim=DIM,
                           max_len=MAXLEN, chunk_words=W, chunk_stride=S,
                           chunk_pool=CHUNK_POOL, topk_docs=TOPK_DOCS,
                           n_docs=N, n_chunks=n, n_queries=len(queries),
                           subcorpus=os.path.basename(a.subcorpus) or None,
                           reranker=None, mmr=None),
               overall=agg(rows),
               by_type={t: dict(n=sum(1 for r in rows if r["qtype"] == t),
                                **agg([r for r in rows if r["qtype"] == t]))
                        for t in sorted({r["qtype"] for r in rows})})
    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump(res, open(f"{OUT}/results/{tag}.json", "w"), indent=2)
    with gzip.open(f"{OUT}/results/{tag}_runs.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"{tag}: {json.dumps(res['overall'])}")
    log("DONE")
    open(f"{WORK}/{tag}.done", "w").write("ok")
