#!/usr/bin/env python3
"""Encode the 13 English NanoBEIR sets with each MoEdernColBERT size (queries and documents).

Writes .work/enc/<set>/<model>_{d,q}.npz (tokens float16 + offsets) and .work/enc/<set>/meta.json.
Resumable: a finished file is skipped. Usage: encode.py [model ...]   (default 1.5M 27M 115M)
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb import MoEdernColBERT

HERE = Path(__file__).resolve().parent
ENC = Path(os.environ.get("MCB_ENC", HERE.parent / ".work" / "enc"))
DEVICE = os.environ.get("MCB_DEVICE", "cpu")
ROOT = os.environ.get("MCB_ROOT")
SETS = ["ArguAna", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO",
        "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact", "Touche2020"]


def load_set(name):
    from datasets import load_dataset
    hf = f"zeta-alpha-ai/Nano{name}"
    corpus = load_dataset(hf, "corpus", split="train")
    queries = load_dataset(hf, "queries", split="train")
    qrels = load_dataset(hf, "qrels", split="train")
    rel = {}
    for r in qrels:
        rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = 1
    qmap = {str(r["_id"]): r["text"] for r in queries}
    qids = [q for q in qmap if q in rel]
    return ([str(r["_id"]) for r in corpus], [r["text"] for r in corpus],
            qids, [qmap[q] for q in qids], {q: rel[q] for q in qids})


def save(path, embs):
    lens = [len(e) for e in embs]
    np.savez(path, tok=np.concatenate(embs).astype(np.float16),
             off=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))


def main(models):
    torch.set_num_threads(4)
    for s in SETS:
        out = ENC / s
        out.mkdir(parents=True, exist_ok=True)
        dids, docs, qids, queries, rel = load_set(s)
        if not (out / "meta.json").exists():
            json.dump(dict(dids=dids, qids=qids, rel=rel), open(out / "meta.json", "w"))
        for name in models:
            m = None
            for kind, texts, task in (("q", queries, "query"), ("d", docs, "document")):
                p = out / f"{name}_{kind}.npz"
                if p.exists():
                    continue
                m = m or MoEdernColBERT(name, device=DEVICE, root=ROOT)
                t0 = time.time()
                E = m.encode(texts, task, batch_size=64 if DEVICE != "cpu" else (16 if name == "115M" else 32))
                save(p, E)
                print(f"[{s}] {name} {kind}: {len(texts)} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["1.5M", "27M", "115M"])
