#!/usr/bin/env python3
"""Encode BEIR Quora (corpus + test queries) with google/embeddinggemma-2, text-only.

Loading, prompts, normalisation and max length come from ../embeddinggemma2-remex-quant/encode.py
(imported by path, not rewritten). fp32, 768d, L2-normalised.

data/ (gitignored):
  quora/corpus.json    {ids, texts}   cached download
  quora/queries.json   test query ids/texts, qrels.json {qid: {did: score}}
  quora/shards/docs_<NNNNN>.npy + docs_<NNNNN>.ids.json   SHARD docs each, corpus order
  quora/q_dense.npy + query_ids.json
Restart resumes at the first missing shard. --limit N = first N docs of a seeded shuffle
(timing run; use --tag to keep it apart from the full run). Log: data/<tag>/encode.log.
"""
from __future__ import annotations
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rq_encode", HERE.parent / "embeddinggemma2-remex-quant" / "encode.py")
rq = importlib.util.module_from_spec(spec); spec.loader.exec_module(rq)


def load_quora(root: Path):
    c = root / "corpus.json"
    if c.exists():
        z = json.load(open(c)); docs, dids = z["texts"], z["ids"]
        q = json.load(open(root / "queries.json")); qrels = json.load(open(root / "qrels.json"))
        return docs, dids, q["texts"], q["ids"], qrels
    from datasets import load_dataset
    corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
    queries = load_dataset("BeIR/quora", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/quora-qrels", split="test")
    qrels = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qmap = {str(r["_id"]): r["text"] for r in queries}
    qids = [q for q in qrels if q in qmap]
    docs = [f"title: {(r['title'] or '').strip() or 'none'} | text: {r['text'].strip()}" for r in corpus]
    dids = [str(r["_id"]) for r in corpus]
    root.mkdir(parents=True, exist_ok=True)
    json.dump({"ids": dids, "texts": docs}, open(c, "w"))
    json.dump({"ids": qids, "texts": [qmap[q] for q in qids]}, open(root / "queries.json", "w"))
    qrels = {q: qrels[q] for q in qids}
    json.dump(qrels, open(root / "qrels.json", "w"))
    return docs, dids, [qmap[q] for q in qids], qids, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="encode only N docs (seeded random subset)")
    ap.add_argument("--qlimit", type=int, default=None, help="encode only N test queries (default: all; 1000 with --limit)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=None, help="stop after the first N shards (corpus-order prefix), then encode queries")
    ap.add_argument("--tag", default=None, help="output subdir under data/ (default: quora, or quora_limitN)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)

    data = HERE / "data"
    docs, dids, queries, qids, qrels = load_quora(data / "quora")
    n_total = len(docs)
    if a.limit:
        sel = np.sort(np.random.default_rng(0).permutation(n_total)[:a.limit])
        docs = [docs[i] for i in sel]; dids = [dids[i] for i in sel]
    qn = a.qlimit or (1000 if a.limit else len(qids))
    queries, qids = queries[:qn], qids[:qn]
    out = data / (a.tag or (f"quora_limit{a.limit}" if a.limit else "quora")); (out / "shards").mkdir(parents=True, exist_ok=True)
    json.dump({q: qrels[q] for q in qids}, open(out / "qrels.json", "w"))
    log = open(out / "encode.log", "a")
    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    m = rq.load_model()
    say(f"docs={len(docs)}/{n_total} queries={len(qids)} threads={a.threads} shard={a.shard} batch={a.batch} model={rq.MODEL}")
    nsh = (len(docs) + a.shard - 1) // a.shard
    if a.max_shards:
        nsh = min(nsh, a.max_shards)
    t0 = time.time(); done_docs = 0
    for s in range(nsh):
        f = out / "shards" / f"docs_{s:05d}.npy"
        lo, hi = s * a.shard, min((s + 1) * a.shard, len(docs))
        if f.exists() and (out / "shards" / f"docs_{s:05d}.ids.json").exists():
            continue
        ts = time.time()
        E = m.encode(docs[lo:hi], batch_size=a.batch, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        json.dump(dids[lo:hi], open(out / "shards" / f"docs_{s:05d}.ids.json", "w"))
        np.save(str(f) + ".tmp.npy", E); Path(str(f) + ".tmp.npy").rename(f)   # npy last: marks shard complete
        done_docs += hi - lo
        rate = done_docs / (time.time() - t0)
        eta = (len(docs) - hi) / rate
        say(f"shard {s+1}/{nsh} docs={hi}/{len(docs)} {(hi-lo)/(time.time()-ts):.1f} docs/s (run avg {rate:.1f}) ETA {eta/3600:.2f} h")
    if not (out / "q_dense.npy").exists():
        tq = time.time()
        Q = m.encode(queries, prompt_name=rq.QUERY_PROMPT, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        np.save(out / "q_dense.npy", Q.astype(np.float32)); json.dump(qids, open(out / "query_ids.json", "w"))
        say(f"queries {len(qids)} in {time.time()-tq:.1f}s ({len(qids)/(time.time()-tq):.1f} q/s)")
    say(f"done total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
