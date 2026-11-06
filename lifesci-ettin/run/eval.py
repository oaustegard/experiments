"""Retrieval eval (CPU, PyTorch) for the arms, MedCPT (vectors from Colab) and BM25.

    python3 -I run/eval.py ARM PARAMS.f32 TOKENIZER [--sets scifact,nfcorpus,pubmed] [--out results/eval_ARM.json]
    python3 -I run/eval.py medcpt   # from .work/teacher/<set>_{docs,queries}.f16.npy
    python3 -I run/eval.py bm25

Docs: title + " " + text (pubmed: the abstract alone, query = its title), at most 512 tokens; queries
carry e17c's query prompt, at most 64 tokens. Exact cosine over the whole corpus; nDCG@10 and recall@100.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from evalset import dense_rank, load, score  # noqa: E402
from model import QP, Embedder, from_flat  # noqa: E402

W = HERE / ".work"
torch.set_num_threads(4)


def sets(names):
    for n in names:
        if n == "pubmed":
            ev = json.loads((W / "data/pubmed_eval.json").read_text())
            docs = {"ids": ev["ids"], "title": [""] * len(ev["ids"]), "text": ev["text"]}
            yield n, docs, {"ids": ev["ids"], "text": ev["title"]}, {i: {i: 1} for i in ev["ids"]}
        else:
            d, q, qr = load(n)
            yield n, d, q, qr


@torch.inference_mode()
def encode(m, tok, texts, max_len, bs=64):
    tok.no_padding()
    tok.enable_truncation(max_len)
    order = np.argsort([len(t) for t in texts])
    pad = tok.token_to_id("[PAD]")
    out = np.zeros((len(texts), 768), np.float32)
    for i in range(0, len(order), bs):
        ix = order[i:i + bs]
        enc = tok.encode_batch([texts[j] for j in ix])
        S = max(len(e.ids) for e in enc)
        ids = torch.full((len(ix), S), pad)
        att = torch.zeros((len(ix), S), dtype=torch.long)
        for b, e in enumerate(enc):
            ids[b, :len(e.ids)] = torch.tensor(e.ids)
            att[b, :len(e.ids)] = 1
        out[ix] = m(ids, att).numpy()
    return out


def run_student(arm, params, tokp, names):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokp))
    flat = np.fromfile(params, np.float32)
    V = (len(flat) - 4099840) // 256   # non-table parameter count of Ettin-17m + dense
    m = Embedder(from_flat(flat, V)).eval()
    res = {}
    for n, d, q, qr in sets(names):
        t0 = time.time()
        D = encode(m, tok, [(a + " " + b).strip() for a, b in zip(d["title"], d["text"])], 512)
        Q = encode(m, tok, [QP + x for x in q["text"]], 64)
        res[n] = score(dense_rank(Q, D, d["ids"]), q["ids"], qr) | {"encode_s": round(time.time() - t0, 1)}
        print(arm, n, round(res[n]["ndcg@10"], 4), round(res[n]["recall@100"], 4), res[n]["encode_s"], flush=True)
    return res


def run_pooled(arm, params, tokp, pooled, name="trec-covid"):
    """Docs pooled on the bridge (web/bridge_encode.js), Dense applied here; queries encoded here."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokp))
    flat = np.fromfile(params, np.float32)
    V = (len(flat) - 4099840) // 256
    p = from_flat(flat, V)
    m = Embedder(p).eval()
    d, q, qr = load(name)
    P = np.load(pooled) if str(pooled).endswith(".npy") else np.fromfile(pooled, np.float32)
    D = P.reshape(len(d["ids"]), 256) @ p["dense"].numpy().T
    Q = encode(m, tok, [QP + x for x in q["text"]], 64)
    r = {name: score(dense_rank(Q, D, d["ids"]), q["ids"], qr) | {"docs_pooled_on": "bridge"}}
    print(arm, name, round(r[name]["ndcg@10"], 4), round(r[name]["recall@100"], 4), flush=True)
    return r


def run_medcpt(names):
    res = {}
    for n, d, q, qr in sets(names):
        D = np.load(W / f"teacher/{n}_docs.f16.npy").astype(np.float32)
        Q = np.load(W / f"teacher/{n}_queries.f16.npy").astype(np.float32)
        # MedCPT scores by inner product, not cosine
        s = Q @ D.T
        top = np.argsort(-s, axis=1)[:, :100]
        res[n] = score([[d["ids"][j] for j in t] for t in top], q["ids"], qr)
        print("medcpt", n, round(res[n]["ndcg@10"], 4), flush=True)
    return res


def run_bm25(names):
    import bm25s
    import Stemmer
    st = Stemmer.Stemmer("english")
    res = {}
    for n, d, q, qr in sets(names):
        corpus = [(a + " " + b).strip() for a, b in zip(d["title"], d["text"])]
        r = bm25s.BM25()
        r.index(bm25s.tokenize(corpus, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
        docs, _ = r.retrieve(bm25s.tokenize(q["text"], stopwords="en", stemmer=st, show_progress=False), k=100, show_progress=False)
        res[n] = score([[d["ids"][j] for j in row] for row in docs], q["ids"], qr)
        print("bm25", n, round(res[n]["ndcg@10"], 4), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("params", nargs="?")
    ap.add_argument("tok", nargs="?")
    ap.add_argument("--sets", default="scifact,nfcorpus,pubmed")
    ap.add_argument("--out")
    ap.add_argument("--pooled", help="docs pooled on the bridge (trec-covid)")
    a = ap.parse_args()
    names = a.sets.split(",")
    r = run_pooled(a.arm, a.params, a.tok, a.pooled) if a.pooled else run_medcpt(names) if a.arm == "medcpt" else run_bm25(names) if a.arm == "bm25" else run_student(a.arm, a.params, a.tok, names)
    out = Path(a.out or HERE / "results" / f"eval_{a.arm}.json")
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps(prev | r, indent=1))
