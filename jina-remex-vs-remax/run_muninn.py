#!/usr/bin/env python3
"""remex (rotation + Lloyd-Max scalar quant) as a practical compressed-Jina
vector format, vs remax 1-bit centered-SimHash — on the muninn corpus.

Embeds the muninn corpus + phase-0 gold queries with the q4 Jina ONNX (the
deployable embedder; its output is fp32-parity float vectors), then scores
recall@5/@10 (chunks collapsed to distinct posts) for:

  * fp32 cosine                         — the embedding ceiling
  * remex {8,4,2}-bit @ d=768           — native recall/byte dial
  * remex 4-bit @ d=512 (256 B)         — byte-matched to remax d512/k4
  * remex two-stage (8-bit, coarse 4)   — Matryoshka coarse->rerank
  * remax 1-bit {d256/k8, d512/k4, d768/k2}

Jina vectors are L2-normalized (norms==1), so remex's per-row fp32 norm is
redundant and EXCLUDED from bytes/row here (sign/IP ranking is unaffected).

n=5 queries: directional only. See run_nfcorpus.py for the 120-query curve.
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
KB = A / "muninn.kb"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(spoke("remax") / "src"))
sys.path.insert(0, str(experiment("lexical-kb-phase0")))
from sweep import QUERIES, stem                       # noqa: E402
from remax import StackedSignBitQuantizer             # noqa: E402
from remax_kb._hamming import hamming_scan, top_k     # noqa: E402
from remex import Quantizer                           # noqa: E402

KS = (5, 10)
SEED = 0


def load_chunks():
    z = zipfile.ZipFile(KB)
    c = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [x["text"] for x in c], [stem(x["meta"]["source_path"]) for x in c]


def embed(emb, texts, tag, dim=768, batch=8):
    """Resumable: memmap + progress counter survive an arena SIGKILL mid-embed."""
    cache = A / f".vec_{tag}.npz"
    if cache.exists():
        m = np.load(cache)["m"]
        if m.shape[0] == len(texts):
            return m
    mm = A / f".vec_{tag}.mm.npy"; prog = A / f".vec_{tag}.prog"; n = len(texts)
    prompt = "document" if tag == "doc" else "query"
    if mm.exists() and prog.exists():
        mat = np.lib.format.open_memmap(mm, mode="r+"); done = int(prog.read_text().strip())
        if mat.shape != (n, dim):
            mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    else:
        mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    t0 = time.time()
    for i in range(done, n, batch):
        v = np.asarray(emb.encode(texts[i:i + batch], prompt=prompt), dtype=np.float32)
        mat[i:i + v.shape[0]] = v; mat.flush(); prog.write_text(str(min(i + batch, n)))
        if (i // batch) % 20 == 0:
            print(f"  {tag} {min(i+batch,n)}/{n}", flush=True)
    out = np.array(mat); np.savez(cache, m=out)
    mm.unlink(missing_ok=True); prog.unlink(missing_ok=True)
    print(f"  embedded {tag} {out.shape} in {time.time()-t0:.0f}s", flush=True)
    return out


def recall_from_order(order, posts, ks):
    """order: doc indices best->worst. Collapse to distinct posts, recall vs gold."""
    agg = {k: 0.0 for k in ks}
    for j, q in enumerate(QUERIES):
        seen, ranked = set(), []
        for i in order[j]:
            if posts[i] not in seen:
                seen.add(posts[i]); ranked.append(posts[i])
            if len(ranked) >= max(ks):
                break
        pos = {p: r + 1 for r, p in enumerate(ranked)}
        gold = set(q["gold"])
        for k in ks:
            agg[k] += sum(1 for g in gold if pos.get(g, 10**9) <= k) / len(gold)
    return {k: agg[k] / len(QUERIES) for k in ks}


def order_floatcos(D, Q):
    return [np.argsort(-(D @ Q[j])) for j in range(len(QUERIES))]


def order_remax(D, Q, dim, k):
    mean = D.mean(0).astype(np.float32)
    qz = StackedSignBitQuantizer(d=dim, k=k, seed=SEED)
    dcodes = qz.encode(np.ascontiguousarray((D - mean)[:, :dim]))
    qcodes = qz.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
    return [top_k(hamming_scan(dcodes, qcodes[j]), len(D)) for j in range(len(QUERIES))]


def order_remex(D, Q, bits, dim=None, twostage=False, coarse=None, cand=200):
    X = D if dim is None else D[:, :dim]
    Xq = Q if dim is None else Q[:, :dim]
    d = X.shape[1]
    qz = Quantizer(d=d, bits=bits, seed=SEED)
    comp = qz.encode(np.ascontiguousarray(X))
    orders = []
    for j in range(len(QUERIES)):
        if twostage:
            idx, _ = qz.search_twostage(comp, Xq[j], k=len(D), candidates=cand, coarse_precision=coarse)
        else:
            idx, _ = qz.search(comp, Xq[j], k=len(D))
        orders.append(idx)
    return orders, qz, comp


def main():
    from remax_kb.embedders import JinaQ4ONNXEmbedder
    emb = JinaQ4ONNXEmbedder(model_path=A / "model.q4.onnx", tokenizer_path=A / "tokenizer.json")
    docs, posts = load_chunks()
    print(f"corpus {len(docs)} chunks / {len(set(posts))} posts; q4 Jina embed", flush=True)
    D = embed(emb, docs, "doc")
    Q = embed(emb, [q["query"] for q in QUERIES], "qry")
    print(f"unit-norm check: doc norms mean={np.linalg.norm(D,axis=1).mean():.4f}\n", flush=True)

    rows = []  # (label, bits, dim, bytes_per_row, r5, r10)
    fc = recall_from_order(order_floatcos(D, Q), posts, KS)
    rows.append(("fp32 cosine", "-", 768, 768 * 4, fc[5], fc[10]))

    for bits, dim in [(8, 768), (4, 768), (2, 768), (4, 512)]:
        orders, _, _ = order_remex(D, Q, bits, dim=dim)
        r = recall_from_order(orders, posts, KS)
        rows.append((f"remex {bits}-bit", bits, dim, dim * bits // 8, r[5], r[10]))

    # two-stage 8-bit, coarse 4-bit
    orders, _, _ = order_remex(D, Q, 8, dim=768, twostage=True, coarse=4)
    r = recall_from_order(orders, posts, KS)
    rows.append(("remex 8b 2stage(c4)", 8, 768, 768 * 8 // 8, r[5], r[10]))

    for dim, k in [(256, 8), (512, 4), (768, 2)]:
        r = recall_from_order(order_remax(D, Q, dim, k), posts, KS)
        rows.append((f"remax 1-bit d{dim}/k{k}", 1, dim, dim * k // 8, r[5], r[10]))

    print("=" * 70)
    print(f"{'method':<22}{'bits':>5}{'dim':>5}{'B/row':>7}{'R@5':>8}{'R@10':>8}")
    print("-" * 70)
    for lab, b, dim, by, r5, r10 in rows:
        print(f"{lab:<22}{str(b):>5}{dim:>5}{by:>7}{r5:>8.3f}{r10:>8.3f}")
    print("-" * 70)
    print("baselines (same gold/corpus): lexical 1.00/1.00 | n=5 directional")
    (A / ".rows_muninn.json").write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
