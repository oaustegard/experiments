"""Dense baseline: gemini-embedding-2 (3072-d) via the CF AI Gateway, cached to emb/*.npy."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

from data import DATA, MAX_CHARS

HERE = Path(__file__).resolve().parent
EMB = HERE / "emb"
MODEL = "gemini-embedding-2"


def embed(texts: list[str], task: str) -> np.ndarray:
    base = (f"https://gateway.ai.cloudflare.com/v1/{os.environ['CF_ACCOUNT_ID']}/{os.environ['CF_GATEWAY_ID']}"
            f"/google-ai-studio/v1beta/models/{MODEL}:batchEmbedContents")
    h = {"Content-Type": "application/json", "cf-aig-authorization": "Bearer " + os.environ["CF_API_TOKEN"]}
    out = []
    for i in range(0, len(texts), 100):
        body = {"requests": [{"model": f"models/{MODEL}", "content": {"parts": [{"text": t}]}, "taskType": task}
                             for t in texts[i:i + 100]]}
        for attempt in range(6):
            r = requests.post(base, headers=h, json=body, timeout=180)
            if r.ok:
                break
            print(f"embed {i}: HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
            time.sleep(2 * 2 ** attempt)
        r.raise_for_status()
        out += [e["values"] for e in r.json()["embeddings"]]
    x = np.asarray(out, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cached(name: str, texts: list[str], task: str) -> np.ndarray:
    EMB.mkdir(exist_ok=True)
    p = EMB / f"{name}.npy"
    if p.exists():
        return np.load(p)
    x = embed(texts, task)
    np.save(p, x)
    return x


def jsonl(p):
    return [json.loads(line) for line in open(p)]


if __name__ == "__main__":
    ax = jsonl(DATA / "arxiv.jsonl")
    print("arxiv", cached("arxiv", [r["text"] for r in ax], "CLASSIFICATION").shape)
    sf = DATA / "scifact"
    docs = jsonl(sf / "corpus.jsonl")
    print("scifact docs", cached("scifact_docs", [f"{r['title']}. {r['text']}"[:MAX_CHARS] for r in docs],
                                 "RETRIEVAL_DOCUMENT").shape)
    qs = jsonl(sf / "queries.jsonl")
    print("scifact queries", cached("scifact_queries", [r["text"] for r in qs], "RETRIEVAL_QUERY").shape)
