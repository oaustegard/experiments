"""Arm B: dense retrieval with bge-base-en-v1.5, the paper's own retriever.

No reranker and no MMR, so this is a floor on their S1, not a reproduction.
`--bench N` times the encoder on N chunks and exits, which is how the full-run
cost was estimated before committing to it.
"""
import argparse, json, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import onnxruntime as ort
from tokenizers import Tokenizer

DATA = "/home/user/erb-data"
WORK = "/home/user/erb-work"
BGE = "/home/user/bge"
OUT = os.path.dirname(os.path.abspath(__file__))
CHUNK_WORDS, CHUNK_STRIDE, MAXLEN = 380, 340, 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def make_session(threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(f"{BGE}/model.onnx", so, providers=["CPUExecutionProvider"])

def load_tokenizer():
    tk = Tokenizer.from_file(f"{BGE}/tokenizer.json")
    tk.enable_truncation(max_length=MAXLEN)
    tk.enable_padding(pad_id=0, pad_token="[PAD]")
    return tk

def encode(sess, tk, texts, batch=16):
    """CLS pooling + L2 norm, which is what bge-*-en-v1.5 is trained for."""
    names = {i.name for i in sess.get_inputs()}
    out = np.empty((len(texts), 768), dtype=np.float16)
    for s in range(0, len(texts), batch):
        enc = tk.encode_batch(texts[s:s + batch])
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        am = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = sess.run(None, feed)[0][:, 0]          # CLS
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        out[s:s + len(enc)] = h.astype(np.float16)
    return out

def chunk_all(titles, contents):
    chunks, owner = [], []
    for i, (t, c) in enumerate(zip(titles, contents)):
        w = (f"{t or ''}\n{c or ''}").split()
        if len(w) <= CHUNK_WORDS:
            parts = [" ".join(w)] if w else [""]
        else:
            parts = [" ".join(w[s:s + CHUNK_WORDS]) for s in range(0, max(1, len(w) - CHUNK_STRIDE + 1), CHUNK_STRIDE)]
        chunks.extend(parts); owner.extend([i] * len(parts))
    return chunks, np.asarray(owner, dtype=np.int32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    tk = load_tokenizer()
    sess = make_session(a.threads)

    if a.bench:
        dt = pq.ParquetFile(f"{DATA}/documents_test.parquet")
        b = next(dt.iter_batches(batch_size=max(64, a.bench))).to_pydict()
        chunks, _ = chunk_all(b["title"], b["content"])
        sample = chunks[:a.bench]
        encode(sess, tk, sample[:a.batch])                       # warm
        t0 = time.time(); encode(sess, tk, sample, batch=a.batch); dt_s = time.time() - t0
        rate = len(sample) / dt_s
        log(f"threads={a.threads} batch={a.batch}: {len(sample)} chunks in {dt_s:.1f}s = {rate:.1f} chunks/s")
        log(f"  -> 2.0M chunks would take {2_000_000/rate/3600:.1f} h")
        sys.exit(0)

    # ---------------------------------------------------------------- full run
    log("reading documents parquet ...")
    dt = pq.read_table(f"{DATA}/documents_test.parquet")
    doc_ids = dt.column("doc_id").to_pylist()
    titles = dt.column("title").to_pylist()
    contents = dt.column("content").to_pylist()
    del dt
    log(f"documents: {len(doc_ids)}")

    chunks, owner = chunk_all(titles, contents)
    del titles, contents
    n = len(chunks)
    log(f"chunks: {n} ({n/len(doc_ids):.2f} per doc)")
    np.save(f"{WORK}/owner.npy", owner)
    with open(f"{WORK}/doc_ids.json", "w") as f:
        json.dump(doc_ids, f)

    path = f"{WORK}/vecs.f16.npy"
    vecs = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 768))
    done_path = f"{WORK}/encoded_upto.txt"
    start = int(open(done_path).read().strip()) if os.path.exists(done_path) else 0
    log(f"resuming at chunk {start}")

    STEP = 20000
    t0 = time.time()
    for s in range(start, n, STEP):
        e = min(s + STEP, n)
        vecs[s:e] = encode(sess, tk, chunks[s:e], batch=a.batch)
        vecs.flush()
        open(done_path, "w").write(str(e))
        el = time.time() - t0
        rate = (e - start) / el
        log(f"encoded {e}/{n}  {rate:.1f} chunks/s  eta {(n-e)/rate/3600:.2f} h")
    log("encoding complete")
