"""Why do the fine-tuned models lose TREC-COVID? Diagnostics on the judged pool.

    python3 -I run/diag_trec.py encode NAME CKPT   # .work/diag/NAME_{D,Q}.npy (35,480 judged docs, 50 queries)
    python3 -I run/diag_trec.py report             # results/diag_trec.json

Pool: every document judged for any TREC-COVID test query (35,480; qrels 0/1/2), ranked per query by cosine.
nDCG@10 on this pool is not the full-corpus number, but every model is scored on the same pool.
Report:
  cross   nDCG@10 with queries from one model and documents from another (all start from e17c, so the spaces
          are comparable): which side carries the loss.
  slices  for each model, where relevant documents land: title-only vs title+abstract, by token length.
  top10   what the model puts in its top 10 instead: share of title-only, mean length, judged-nonrelevant.
  geometry mean cosine between random documents (how crowded the space is) and query-doc margins.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from evalset import load  # noqa: E402

W = HERE / ".work"
O = W / "diag"
MODELS = ["e17c", "k0-stock", "k4-mq", "k5-anchor"]


def pool():
    from huggingface_hub import hf_hub_download
    d, q, _ = load("trec-covid")
    qr = {}
    with open(hf_hub_download("BeIR/trec-covid-qrels", "test.tsv", repo_type="dataset")) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for qid, did, s in r:
            qr.setdefault(str(qid), {})[str(did)] = int(s)
    judged = sorted({x for v in qr.values() for x in v})
    ix = {k: i for i, k in enumerate(d["ids"])}
    keep = [ix[x] for x in judged if x in ix]
    docs = {"ids": [d["ids"][i] for i in keep], "title": [d["title"][i] for i in keep], "text": [d["text"][i] for i in keep]}
    return docs, q, qr


def encode(name, ckpt):
    import torch
    from tokenizers import Tokenizer
    from eval import encode as enc
    from model import QP, Embedder, from_flat
    torch.set_num_threads(4)
    docs, q, _ = pool()
    flat = np.fromfile(ckpt, np.float32)
    m = Embedder(from_flat(flat, (len(flat) - 4099840) // 256)).eval()
    tok = Tokenizer.from_file(str(W / "e17c/epoch1/tokenizer.json"))
    O.mkdir(parents=True, exist_ok=True)
    np.save(O / f"{name}_Q.npy", enc(m, tok, [QP + x for x in q["text"]], 64))
    np.save(O / f"{name}_D.npy", enc(m, tok, [(a + " " + b).strip() for a, b in zip(docs["title"], docs["text"])], 512))


def ndcg(rank, rel, k=10):
    dcg = sum((2 ** max(rel.get(d, 0), 0) - 1) / np.log2(i + 2) for i, d in enumerate(rank[:k]))
    ideal = sorted([max(v, 0) for v in rel.values()], reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def report():
    from tokenizers import Tokenizer
    docs, q, qr = pool()
    ids = np.array(docs["ids"])
    tok = Tokenizer.from_file(str(W / "e17c/epoch1/tokenizer.json"))
    tok.no_truncation()
    text = [(a + " " + b).strip() for a, b in zip(docs["title"], docs["text"])]
    ntok = np.array([len(e.ids) for i in range(0, len(text), 1000) for e in tok.encode_batch(text[i:i + 1000])])
    print('tokenized', flush=True)
    title_only = np.array([len(b.strip()) == 0 for b in docs["text"]])
    pos = {k: i for i, k in enumerate(docs["ids"])}
    nz = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
    Q = {m: nz(np.load(O / f"{m}_Q.npy")) for m in MODELS}
    D = {m: nz(np.load(O / f"{m}_D.npy")) for m in MODELS}
    qids = q["ids"]
    out = {"pool_docs": len(ids), "title_only_frac": float(title_only.mean()), "cross": {}, "per_model": {}}
    for qm in MODELS:
        for dm in MODELS:
            S = Q[qm] @ D[dm].T
            out["cross"][f"Q:{qm} D:{dm}"] = round(float(np.mean([ndcg(list(ids[np.argsort(-S[i])[:10]]), qr[qid]) for i, qid in enumerate(qids)])), 4)
    rng = np.random.default_rng(0)
    samp = rng.choice(len(ids), 2000, replace=False)
    lbins = [(0, 32), (32, 128), (128, 256), (256, 512), (512, 10 ** 6)]
    for m in MODELS:
        S = Q[m] @ D[m].T
        r = {}
        G = D[m][samp]
        C = G @ G.T
        r["doc_doc_mean_cos"] = float(C[np.triu_indices(len(samp), 1)].mean())
        top = np.argsort(-S, axis=1)[:, :10]
        r["top10_title_only"] = float(title_only[top].mean())
        r["top10_mean_tokens"] = float(ntok[top].mean())
        lab = np.array([[qr[qid].get(ids[j], None) for j in row] for qid, row in zip(qids, top)], dtype=object)
        r["top10_judged_nonrel"] = float(np.mean([x == 0 for x in lab.ravel()]))
        r["top10_rel"] = float(np.mean([x is not None and x > 0 for x in lab.ravel()]))
        # margin: mean score of relevant (2) docs minus non-relevant (0) docs, per slice, averaged over queries
        sl = {}
        for name, mask in [("title_only", title_only), ("with_abstract", ~title_only)] + [(f"tok{a}-{b}", (ntok >= a) & (ntok < b)) for a, b in lbins]:
            mg, rk = [], []
            for i, qid in enumerate(qids):
                rel = [pos[d] for d, s in qr[qid].items() if s == 2 and d in pos and mask[pos[d]]]
                non = [pos[d] for d, s in qr[qid].items() if s == 0 and d in pos]
                if not rel or not non:
                    continue
                mg.append(S[i, rel].mean() - S[i, non].mean())
                order = np.argsort(-S[i])
                inv = np.empty(len(order), int)
                inv[order] = np.arange(len(order))
                rk.append(np.median(inv[rel]))
            sl[name] = {"n_docs": int(mask.sum()), "margin": round(float(np.mean(mg)), 4), "median_rank_of_rel": round(float(np.mean(rk)), 1)}
        r["slices"] = sl
        r["ndcg10_pool"] = out["cross"][f"Q:{m} D:{m}"]
        out["per_model"][m] = r
    (HERE / "results/diag_trec.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"encode": lambda: encode(sys.argv[2], sys.argv[3]), "report": report}[sys.argv[1]]()
