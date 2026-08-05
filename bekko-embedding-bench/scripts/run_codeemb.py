"""Does a code-trained encoder beat a general one on NL->code file discovery?

Part A is cross-modal: a natural-language bug report retrieving source files.
bekko is trained for text retrieval; jina-embeddings-v2-base-code is trained on
(docstring, code) pairs across 30 languages, which is precisely that pairing.
Same n=59 instances, same AST corpus, same grep baseline — only the encoder
changes.

Also fuses each dense arm with grep by RRF, because every previous round of this
experiment found the two fail on disjoint instances.
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder  # noqa: E402
from eval_search import (  # noqa: E402
    approx_tokens, arm_dense, arm_rg, extract_identifiers, recall_at, rrf,
)
from jinacode import JinaCodeEncoder  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def sign_test(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    d = a - b
    w, l = int((d > 0).sum()), int((d < 0).sum())
    n = w + l
    if n == 0:
        return w, l, 1.0
    k = min(w, l)
    return w, l, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def boot(a: np.ndarray, b: np.ndarray, reps: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    d = (a - b)[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    inst = json.load(open(HERE / "instances.json"))
    chunks = json.load(open(HERE / "chunks_ast.json"))
    n = len(chunks)

    # grep baseline, shared by every arm
    rg_cache = {}
    for it in inst:
        q = it["title"] + "\n" + it["body"]
        ranked, wall, chars = arm_rg(extract_identifiers(q))
        rg_cache[it["issue"]] = (ranked, wall, chars)

    encoders = {
        "bekko-a8m": (lambda: BekkoEncoder("a8m", threads=4), "vecs_ast_a8m.f32", 384),
        "bekko-a25m": (lambda: BekkoEncoder("a25m", threads=4), "vecs_ast_a25m.f32", 384),
        "jina-code": (lambda: JinaCodeEncoder(threads=4), "vecs_ast_jinacode.f32", 768),
    }

    per = {}
    rows = []
    for name, (make, vecfile, dim) in encoders.items():
        p = HERE / vecfile
        done = HERE / (vecfile.replace(".f32", ".done"))
        if not p.exists() or (done.exists() and int(done.read_text()) < n):
            print(f"SKIP {name}: corpus not fully encoded", flush=True)
            continue
        mat = np.asarray(np.memmap(p, dtype=np.float32, mode="r", shape=(n, dim)))
        enc = make()
        d5 = np.zeros(len(inst)); d10 = np.zeros(len(inst))
        f5 = np.zeros(len(inst)); f10 = np.zeros(len(inst))
        g5 = np.zeros(len(inst)); g10 = np.zeros(len(inst))
        toks = 0
        for i, it in enumerate(inst):
            q = it["title"] + "\n" + it["body"]
            qv = enc.encode([q], sort_by_length=False)[0]
            dranked, backing = arm_dense(qv, mat, chunks)
            rranked, _, _ = rg_cache[it["issue"]]
            fused = rrf(rranked, dranked)
            gold = it["gold"]
            d5[i], d10[i] = recall_at(dranked, gold, 5), recall_at(dranked, gold, 10)
            f5[i], f10[i] = recall_at(fused, gold, 5), recall_at(fused, gold, 10)
            g5[i], g10[i] = recall_at(rranked, gold, 5), recall_at(rranked, gold, 10)
            toks += approx_tokens(sum(len(chunks[j]["text"]) for j in backing[:10]))
        per[name] = {"d5": d5, "d10": d10, "f5": f5, "f10": f10, "g5": g5, "g10": g10}
        rows.append({"model": name, "dim": dim,
                     "dense_r5": d5.mean(), "dense_r10": d10.mean(),
                     "rrf_r5": f5.mean(), "rrf_r10": f10.mean(),
                     "rg_r5": g5.mean(), "rg_r10": g10.mean(), "tokens": toks})
        print(f"{name:12s} dense {d5.mean():.3f}/{d10.mean():.3f}  "
              f"rrf {f5.mean():.3f}/{f10.mean():.3f}  (rg {g5.mean():.3f}/{g10.mean():.3f})",
              flush=True)
        del mat, enc

    json.dump(rows, open(HERE / "results_codeemb.json", "w"), indent=1,
              default=float)

    if "jina-code" in per:
        print("\n=== paired tests, n=%d ===" % len(inst))
        print("%-52s %8s %18s %8s %8s  %s"
              % ("comparison", "delta", "95% CI", "w/l", "p", "verdict"))
        cmps = [("jina-code dense beats rg (r@5)", per["jina-code"]["d5"], per["jina-code"]["g5"]),
                ("jina-code dense beats rg (r@10)", per["jina-code"]["d10"], per["jina-code"]["g10"]),
                ("jina-code beats bekko-a25m (dense r@5)", per["jina-code"]["d5"], per["bekko-a25m"]["d5"]),
                ("jina-code beats bekko-a8m (dense r@5)", per["jina-code"]["d5"], per["bekko-a8m"]["d5"]),
                ("jina-code beats bekko-a25m (dense r@10)", per["jina-code"]["d10"], per["bekko-a25m"]["d10"]),
                ("jina-code RRF beats rg (r@10)", per["jina-code"]["f10"], per["jina-code"]["g10"]),
                ("jina-code RRF beats bekko RRF (r@10)", per["jina-code"]["f10"], per["bekko-a25m"]["f10"])]
        for lbl, a, b in cmps:
            w, l, p = sign_test(a, b)
            lo, hi = boot(a, b)
            print("%-52s %+8.3f  [%+.3f,%+.3f] %4d/%-3d %8.4f  %s"
                  % (lbl, a.mean() - b.mean(), lo, hi, w, l, p,
                     "SUPPORTED" if p < 0.05 else "noise"))


if __name__ == "__main__":
    main()
