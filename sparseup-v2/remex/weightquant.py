"""Step 3: effect of fake-quantized model weights on retrieval and on the sparse vectors.

Reads the Kaggle T4 encodes in cache/kaggle_wq/ (fp32 weights, int8 per-output-row
symmetric, int4 per-row group-32 symmetric; all Linear weights + the token embedding,
which is tied to the MLM decoder) and the local CPU bf16 encodes (the card's default
precision) for context. Writes results/weightquant.json.
"""
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import compress as C

HERE = Path(__file__).resolve().parent
WQ = HERE / "cache" / "kaggle_wq"


def load_variant(name, variant):
    D = sp.load_npz(WQ / f"{name}_{variant}_docs.npz").tocsr()
    Q = sp.load_npz(WQ / f"{name}_{variant}_queries.npz").tocsr()
    meta = json.loads((WQ / f"{name}_{variant}_meta.json").read_text())
    return D, Q, meta


def row_cos(X, Y):
    num = np.asarray(X.multiply(Y).sum(1)).ravel()
    den = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel() * np.asarray(Y.multiply(Y).sum(1)).ravel())
    return num / np.maximum(den, 1e-12)


def topk_overlap(X, Y, k=32):
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        a = X.indices[X.indptr[i]:X.indptr[i + 1]][np.argsort(-X.data[X.indptr[i]:X.indptr[i + 1]])[:k]]
        b = Y.indices[Y.indptr[i]:Y.indptr[i + 1]][np.argsort(-Y.data[Y.indptr[i]:Y.indptr[i + 1]])[:k]]
        out[i] = len(set(a.tolist()) & set(b.tolist())) / max(min(k, len(a)), 1)
    return out


def evaluate(D, Q, meta):
    qi, ideal, ex, npos = C.prep_qrels(meta)
    top = C.score_topn(Q, D, exclude=ex, tiekey=C.tie_key(meta["dids"]))
    return C.metrics(top, qi, ideal, npos)


def main():
    out = {"note": "fp32-compute fake quantization on a Tesla T4; reference = fp32 weights on the same GPU. "
                   "bf16_cpu = the card's default bf16 weights on this container's CPU (the step-1 baseline).",
           "datasets": {}}
    for name in ["scifact", "nfcorpus"]:
        Dr, Qr, mr = load_variant(name, "fp32")
        nd_r, rc_r = evaluate(Dr, Qr, mr)
        res = {"fp32": {"ndcg10": float(nd_r.mean()), "r100": float(rc_r.mean()),
                        "doc_nnz": float(np.diff(Dr.indptr).mean()), "query_nnz": float(np.diff(Qr.indptr).mean())}}
        variants = [("int8", load_variant(name, "int8")), ("int4g32", load_variant(name, "int4g32"))]
        Db, Qb, mb = C.load(name)
        assert mb["dids"] == mr["dids"] and mb["qids"] == mr["qids"]
        variants.append(("bf16_cpu", (Db, Qb, mb)))
        for v, (D, Q, m) in variants:
            assert m["dids"] == mr["dids"]
            nd, rc = evaluate(D, Q, m)
            dn, dr = nd - nd_r, rc - rc_r
            res[v] = {"ndcg10": float(nd.mean()), "r100": float(rc.mean()),
                      "d_ndcg10": float(dn.mean()), "d_ndcg10_ci": C.boot_ci(dn),
                      "d_r100": float(dr.mean()), "d_r100_ci": C.boot_ci(dr),
                      "doc_cos_mean": float(row_cos(D, Dr).mean()), "query_cos_mean": float(row_cos(Q, Qr).mean()),
                      "doc_top32_overlap": float(topk_overlap(D, Dr).mean()),
                      "query_top32_overlap": float(topk_overlap(Q, Qr).mean()),
                      "doc_nnz": float(np.diff(D.indptr).mean()), "query_nnz": float(np.diff(Q.indptr).mean())}
            print(name, v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res[v].items()}, flush=True)
        res["n_queries"] = len(nd_r)
        res["n_docs"] = Dr.shape[0]
        out["datasets"][name] = res
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "weightquant.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
