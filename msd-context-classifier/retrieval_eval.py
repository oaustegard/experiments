"""Question -> product-page retrieval over every product page in the full crawl.

Usage: python3 retrieval_eval.py --arm bm25 | --arm encoder --model <dir> --name <name> [--pool cls|mean]
Index documents = title + text (truncated to --max-len tokens for encoders) of every 'products' page in
data/full/corpus.jsonl (falls back to data/corpus.jsonl). Queries = data/product_queries.jsonl.
Metrics: recall@1, recall@10, MRR, overall and split by seen/unseen (split field) and has_code.
Encoder arms: mean-pooled (or CLS) last hidden state, L2-normalised, cosine. Document embeddings are cached
per model dir under results/cache/. Writes results/retr_<name>.json. Completion line: 'done -> <path>'."""
import argparse, json, os, sys, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_corpus, log

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["bm25", "encoder"], required=True); ap.add_argument("--model"); ap.add_argument("--name")
ap.add_argument("--pool", default="mean"); ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
corpus = os.path.join(HERE, "data", "full", "corpus.jsonl")
if not os.path.exists(corpus): corpus = os.path.join(HERE, "data", "corpus.jsonl")
docs = [r for r in load_corpus(corpus) if r["label"] == "products"]
urls = [d["url"] for d in docs]; uidx = {u: i for i, u in enumerate(urls)}
texts = [((d.get("title") or "") + "\n" + (d.get("text") or "")).strip() for d in docs]
qs = [json.loads(l) for l in open(os.path.join(HERE, "data", "product_queries.jsonl")) if l.strip()]
qs = [q for q in qs if q["url"] in uidx]
log(f"index {len(docs)} product pages from {corpus}; {len(qs)} queries with a target in the index")
name = a.name or a.arm
t0 = time.time()
if a.arm == "bm25":
    import bm25s
    tokd = bm25s.tokenize(texts, stopwords="en"); ret = bm25s.BM25(); ret.index(tokd)
    res_idx, _ = ret.retrieve(bm25s.tokenize([q["text"] for q in qs], stopwords="en"), k=min(100, len(docs)))
    ranks = [list(res_idx[i]) for i in range(len(qs))]
else:
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(a.model); enc = AutoModel.from_pretrained(a.model); enc.eval()
    def emb(batch_texts):
        out = []
        with torch.no_grad():
            for i in range(0, len(batch_texts), a.bs):
                e = tok(batch_texts[i:i + a.bs], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
                h = enc(**{k: v for k, v in e.items() if k in ("input_ids", "attention_mask")}).last_hidden_state
                if a.pool == "cls": v = h[:, 0]
                else:
                    m = e["attention_mask"].unsqueeze(-1).float(); v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(v, dim=-1))
        return torch.cat(out).numpy()
    key = hashlib.sha1((os.path.abspath(a.model) + a.pool + str(a.max_len) + corpus + str(len(docs))).encode()).hexdigest()[:12]
    cache = os.path.join(HERE, "results", "cache", f"docs_{key}.npy"); os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache): D = np.load(cache); log("doc embeddings from cache")
    else:
        D = emb(texts); np.save(cache, D); log(f"embedded {len(texts)} docs in {time.time()-t0:.0f}s")
    Q = emb([q["text"] for q in qs]); S = Q @ D.T
    ranks = [list(np.argsort(-S[i])[:100]) for i in range(len(qs))]

def metrics(sel):
    r1 = r10 = mrr = 0.0
    for i in sel:
        g = uidx[qs[i]["url"]]; rk = ranks[i]
        pos = rk.index(g) + 1 if g in rk else None
        r1 += pos == 1; r10 += bool(pos and pos <= 10); mrr += (1 / pos) if pos else 0.0
    n = max(1, len(sel)); return {"n": len(sel), "recall@1": r1 / n, "recall@10": r10 / n, "mrr": mrr / n}
res = {"name": name, "arm": a.arm, "model": a.model, "pool": a.pool, "n_docs": len(docs), "seconds": time.time() - t0, "all": metrics(range(len(qs)))}
for k, f in [("unseen", lambda q: q["split"] != "train"), ("seen", lambda q: q["split"] == "train"), ("with_code", lambda q: q.get("has_code")), ("no_code", lambda q: not q.get("has_code"))]:
    res[k] = metrics([i for i, q in enumerate(qs) if f(q)])
out = os.path.join(HERE, "results", f"retr_{name}.json"); json.dump(res, open(out, "w"), indent=1)
log(f"{name}: R@1 {res['all']['recall@1']:.3f} R@10 {res['all']['recall@10']:.3f} MRR {res['all']['mrr']:.3f} | unseen R@10 {res['unseen']['recall@10']:.3f} | code R@10 {res['with_code']['recall@10']:.3f} no-code {res['no_code']['recall@10']:.3f}")
print(f"done -> {out}")
