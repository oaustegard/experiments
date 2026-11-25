"""Parity fixture: 50 texts (20 queries with [Q], 30 documents with [D], 5 to 512 tokens), their token ids
and pooling masks, and the reference vectors the WGSL engine is compared against.

    python3 fixture.py          # web/fixtures/parity.json (texts, ids, pool) + cache/parity_ref.npz

Reference arms, all PyTorch CPU on the same ids: fp32; bf16 (the model's default dtype, for scale);
int8 and q4 (fp32 compute on weights dequantized from the pack bytes, so quantization error is separate
from kernel error).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pack import dequant_state_dict
from ref import Reference

HERE = Path(__file__).resolve().parent

QUERIES = [
    "what is a cat", "how do vaccines train the immune system", "best budget laptop for programming 2026",
    "symptoms of vitamin D deficiency", "who wrote the declaration of independence", "SPLADE sparse retrieval",
    "can dogs eat grapes", "rust borrow checker lifetime error",
]


def texts(tok) -> list[tuple[str, str, str]]:
    """-> [(name, kind, text)]"""
    d = HERE / "cache" / "scifact"
    corpus = pd.read_parquet(d / "corpus" / "corpus-00000-of-00001.parquet")
    queries = pd.read_parquet(d / "queries" / "queries-00000-of-00001.parquet")
    qrels = pd.read_csv(d / "test.tsv", sep="\t", dtype=str)
    out = [(f"q{i}", "query", q) for i, q in enumerate(QUERIES)]
    test_q = queries[queries["_id"].astype(str).isin(set(qrels["query-id"]))]
    out += [(f"sq{i}", "query", t) for i, t in zip(test_q["_id"].head(12), test_q["text"].head(12))]
    docs = [(str(i), (a + " " + b).strip()) for i, a, b in zip(corpus["_id"], corpus["title"], corpus["text"])]
    lens = np.array([len(tok(t)["input_ids"]) for _, t in docs[:2000]])
    rng = np.random.default_rng(0)
    picked = []
    for lo, hi, n in [(0, 64, 4), (64, 160, 6), (160, 300, 8), (300, 520, 6)]:
        idx = np.flatnonzero((lens >= lo) & (lens < hi))
        picked += [docs[i] for i in rng.choice(idx, size=min(n, len(idx)), replace=False)]
    out += [(f"d{i}", "document", t) for i, t in picked]
    # Over-long documents: three abstracts joined, truncated to 512 tokens by the tokenizer.
    for k in range(6):
        j = rng.choice(len(docs), size=3, replace=False)
        out.append((f"long{k}", "document", " ".join(docs[i][1] for i in j)))
    out.append(("title-only", "document", docs[7][1].split(".")[0]))
    return out


def sparse(v: torch.Tensor) -> dict:
    nz = torch.nonzero(v).flatten()
    return {"i": nz.tolist(), "w": v[nz].tolist()}


def main():
    ref = Reference(HERE / "hf", threads=4)
    rows = []
    for name, kind, text in texts(ref.tok):
        ids, pool = ref.tokenize(text, kind, 512)
        rows.append({"name": name, "kind": kind, "text": text, "ids": ids, "pool": pool, "n": len(ids)})
    arms = {}
    arms["fp32"] = torch.stack([ref.encode_ids(r["ids"], r["pool"]) for r in rows])
    fp32_sd = {k: v.clone() for k, v in ref.state_dict().items()}
    for q in ("int8", "q4"):
        ref.load_weights(dequant_state_dict(HERE / "packs" / q))
        arms[q] = torch.stack([ref.encode_ids(r["ids"], r["pool"]) for r in rows])
    ref.load_weights(fp32_sd)
    ref.model.to(torch.bfloat16)
    arms["bf16"] = torch.stack([ref.encode_ids(r["ids"], r["pool"]) for r in rows])
    (HERE / "web" / "fixtures").mkdir(parents=True, exist_ok=True)
    (HERE / "web" / "fixtures" / "parity.json").write_text(json.dumps(rows))
    np.savez_compressed(HERE / "cache" / "parity_ref.npz", **{k: v.numpy() for k, v in arms.items()})
    lens = [r["n"] for r in rows]
    print(len(rows), "texts; tokens min", min(lens), "max", max(lens), "at 512:", sum(n == 512 for n in lens))


if __name__ == "__main__":
    main()
