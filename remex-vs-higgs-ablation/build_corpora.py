#!/usr/bin/env python3
"""Build the three evaluation corpora as cached fp32 doc/query matrices.

Three sets, deliberately differing in dimensionality and anisotropy, because
the ablation's axis-C prediction (scalar vs vector codebook) is a statement
about the *marginal geometry* of the rotated coordinates and must not be read
off a single spectrum shape:

  arxiv768    750 arXiv abstracts (docs) / 150 titles (queries), d=768,
              BAAI/bge-base-en-v1.5.  Continuity with the prior remex work in
              this repo, which used a 750-abstract arXiv set.
  glove100    ANN-benchmarks glove-100-angular, d=100.  External
              comparability; strongly anisotropic word-vector geometry.
  nfcorpus1024  BEIR NFCorpus medical abstracts, d=1024,
              BAAI/bge-large-en-v1.5.  The deployment-shaped target, and the
              corpus where per-tensor int8 was previously shown domain-fragile
              (METHODS.md, jina-int8-remax_kb).

Everything is cached to assets/<name>.npz and the script is idempotent — a
present, well-formed cache is skipped.  CCotw reaps idle background jobs
(METHODS.md), so the encoder loop mini-batches and checkpoints per corpus.
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.pipeline import retry

ASSETS = HERE / "assets"
DATA = HERE / "data"
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

N_ARXIV_DOCS = 750
N_ARXIV_QUERIES = 150
N_GLOVE_DOCS = 20_000
N_GLOVE_QUERIES = 1_000
BATCH = 16
MAXLEN = 256


def _encode(texts, model_name, maxlen=MAXLEN):
    """Mini-batched encode.  One-shot encode(all) OOMs at ~26GB on the
    attention-mask Expand broadcast for a few-thousand-doc corpus
    (METHODS.md, lfm25-embedder-remax_kb)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = maxlen
    out = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        out.append(model.encode(texts[i : i + BATCH], convert_to_numpy=True,
                                normalize_embeddings=False, show_progress_bar=False))
        if i % (BATCH * 20) == 0:
            done = min(i + BATCH, len(texts))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done}/{len(texts)} @ {rate:.1f}/s", flush=True)
    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------
# arXiv


def _arxiv_fetch(categories, want):
    """Title + abstract pairs, arXiv Atom API first, HF mirror as fallback.

    The live API 429s through this container's egress proxy (see RESULTS.md),
    so the mirror is the path that actually runs here; the API branch is kept
    because it is the source of record when it is reachable.
    """
    cache = DATA / "arxiv_raw.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        if len(rows) >= want:
            return rows[:want]
    rows = _arxiv_from_api(categories, want) or _arxiv_from_hf(want)
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows[:want]


def _arxiv_from_api(categories, want):
    import httpx

    rows, seen = [], set()
    for cat in categories:
        start = 0
        while len(rows) < want and start < 2000:
            params = {"search_query": f"cat:{cat}", "start": start,
                      "max_results": 100, "sortBy": "submittedDate",
                      "sortOrder": "descending"}

            def _get(params=params):
                r = httpx.get(ARXIV_API, params=params, timeout=60,
                              follow_redirects=True)
                r.raise_for_status()
                return r.text

            try:
                root = ET.fromstring(retry(_get, attempts=3, base=3.0))
            except Exception as e:  # noqa: BLE001 — any failure means fall back
                print(f"  [arxiv] API unusable ({type(e).__name__}), using mirror")
                return []
            entries = root.findall("a:entry", NS)
            if not entries:
                break
            for e in entries:
                title = " ".join((e.findtext("a:title", "", NS) or "").split())
                summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
                if len(summary) < 200 or title in seen:
                    continue
                seen.add(title)
                rows.append({"title": title, "abstract": summary})
            start += 100
            time.sleep(3.0)  # arXiv API asks for >=3s between requests
    return rows


def _arxiv_from_hf(want):
    """cs.LG / stat.ML abstracts from the CShorten/ML-ArXiv-Papers mirror."""
    import csv

    from huggingface_hub import hf_hub_download

    p = hf_hub_download("CShorten/ML-ArXiv-Papers", "ML-Arxiv-Papers.csv",
                        repo_type="dataset")
    rows, seen = [], set()
    with open(p, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            title = " ".join((rec.get("title") or "").split())
            abstract = " ".join((rec.get("abstract") or "").split())
            if len(abstract) < 400 or not title or title in seen:
                continue
            seen.add(title)
            rows.append({"title": title, "abstract": abstract})
            if len(rows) >= want * 3:
                break
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(want, len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


def build_arxiv():
    dst = ASSETS / "arxiv768.npz"
    if dst.exists():
        print("[arxiv768] cached, skip")
        return
    rows = _arxiv_fetch(["cs.LG", "stat.ML", "cs.CL", "cs.IR"], N_ARXIV_DOCS)
    if len(rows) < N_ARXIV_DOCS:
        raise SystemExit(f"[arxiv768] only got {len(rows)} abstracts")
    print(f"[arxiv768] {len(rows)} abstracts; encoding with bge-base", flush=True)
    docs = _encode([r["abstract"] for r in rows], "BAAI/bge-base-en-v1.5")
    rng = np.random.default_rng(0)
    qidx = rng.choice(len(rows), size=N_ARXIV_QUERIES, replace=False)
    # bge wants the retrieval instruction prefix on the query side only
    qtexts = ["Represent this sentence for searching relevant passages: "
              + rows[i]["title"] for i in qidx]
    qry = _encode(qtexts, "BAAI/bge-base-en-v1.5", maxlen=64)
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, docs=docs, queries=qry, qgold=qidx.astype(np.int32))
    print(f"[arxiv768] docs {docs.shape} queries {qry.shape}")


# --------------------------------------------------------------------------
# GloVe (ANN-benchmarks)


def build_glove():
    dst = ASSETS / "glove100.npz"
    if dst.exists():
        print("[glove100] cached, skip")
        return
    import h5py

    src = ASSETS / "glove.hdf5"
    if not src.exists():
        raise SystemExit("[glove100] assets/glove.hdf5 missing — "
                         "curl -L -o assets/glove.hdf5 "
                         "http://ann-benchmarks.com/glove-100-angular.hdf5")
    with h5py.File(src, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
    rng = np.random.default_rng(0)
    didx = rng.choice(train.shape[0], size=N_GLOVE_DOCS, replace=False)
    docs = np.ascontiguousarray(train[np.sort(didx)])
    qry = np.ascontiguousarray(test[:N_GLOVE_QUERIES])
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, docs=docs, queries=qry)
    print(f"[glove100] docs {docs.shape} queries {qry.shape}")


# --------------------------------------------------------------------------
# NFCorpus (BEIR)


def _nfcorpus_files():
    """BEIR NFCorpus, from the mteb HF mirror (the UKP zip host times out here)."""
    from huggingface_hub import hf_hub_download

    corpus = hf_hub_download("mteb/nfcorpus", "corpus.jsonl", repo_type="dataset")
    queries = hf_hub_download("mteb/nfcorpus", "queries.jsonl", repo_type="dataset")
    return Path(corpus), Path(queries)


def build_nfcorpus():
    dst = ASSETS / "nfcorpus1024.npz"
    if dst.exists():
        print("[nfcorpus1024] cached, skip")
        return
    cpath, qpath = _nfcorpus_files()
    docs_txt = []
    for line in cpath.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            docs_txt.append((o.get("title", "") + " " + o.get("text", "")).strip())
    qtext = []
    for line in qpath.read_text().splitlines():
        if line.strip():
            qtext.append(json.loads(line)["text"])
    qtext = qtext[:400]
    print(f"[nfcorpus1024] {len(docs_txt)} docs / {len(qtext)} queries; "
          f"encoding with bge-large", flush=True)
    docs = _encode(docs_txt, "BAAI/bge-large-en-v1.5")
    qry = _encode(["Represent this sentence for searching relevant passages: " + q
                   for q in qtext], "BAAI/bge-large-en-v1.5", maxlen=64)
    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, docs=docs, queries=qry)
    print(f"[nfcorpus1024] docs {docs.shape} queries {qry.shape}")


def main():
    which = sys.argv[1:] or ["glove", "arxiv", "nfcorpus"]
    for name in which:
        {"glove": build_glove, "arxiv": build_arxiv,
         "nfcorpus": build_nfcorpus}[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
