#!/usr/bin/env python3
"""Group memories into topics sized for one Haiku writer each.

Embeds tags + text with a static model (model2vec potion-base-32M: no GPU, a
few seconds for the whole store), clusters with Ward linkage, then splits any
topic whose text exceeds MAX_CHARS and folds topics under MIN_CHARS into their
nearest neighbour. Confidential and health memories are left out of the codex.

    python3 cluster.py DATA [--k 160]     # reads DATA/memories.jsonl, writes DATA/topics.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

EXCLUDE = {"confidential", "health"}
MAX_CHARS = 110_000
MIN_CHARS = 12_000


def tags(m: dict) -> list[str]:
    t = m.get("tags")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except json.JSONDecodeError:
            t = [t]
    return [str(x) for x in t or []]


def load(data: Path) -> tuple[list[dict], int]:
    ms = [json.loads(line) for line in (data / "memories.jsonl").open()]
    kept = [m for m in ms if not EXCLUDE & set(tags(m)) and (m.get("summary") or "").strip()]
    return kept, len(ms) - len(kept)


def embed(ms: list[dict]) -> np.ndarray:
    from model2vec import StaticModel
    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    texts = [" ".join(t for t in tags(m) if not t.startswith("audit")) + "\n" + m["summary"][:3000] for m in ms]
    v = model.encode(texts)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sizes(labels, chars) -> dict[int, int]:
    s = collections.Counter()
    for lab, c in zip(labels, chars):
        s[lab] += c
    return s


def rebalance(v: np.ndarray, labels: np.ndarray, chars: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    nxt = labels.max() + 1
    changed = True
    while changed:  # split oversize topics in two until all fit
        changed = False
        for lab, size in list(sizes(labels, chars).items()):
            idx = np.where(labels == lab)[0]
            if size > MAX_CHARS and len(idx) > 1:
                sub = KMeans(2, n_init=4, random_state=0).fit_predict(v[idx])
                if sub.min() == sub.max():
                    sub = np.arange(len(idx)) % 2
                labels[idx[sub == 1]] = nxt
                nxt += 1
                changed = True
    while True:  # fold the smallest undersize topic into its nearest neighbour that has room
        s = sizes(labels, chars)
        small = sorted((size, lab) for lab, size in s.items() if size < MIN_CHARS)
        if not small:
            break
        _, lab = small[0]
        cent = {l: v[labels == l].mean(0) for l in s}
        order = sorted((l for l in s if l != lab), key=lambda l: -float(cent[lab] @ cent[l]))
        target = next((l for l in order if s[l] + s[lab] <= MAX_CHARS), None)
        if target is None:
            break
        labels[labels == lab] = target
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--k", type=int, default=160)
    a = ap.parse_args()
    ms, dropped = load(a.data)
    v = embed(ms)
    np.save(a.data / "mem_vecs.npy", v)
    chars = np.array([len(m["summary"]) for m in ms])
    labels = AgglomerativeClustering(n_clusters=a.k, linkage="ward").fit_predict(v)
    labels = rebalance(v, labels, chars)
    topics = []
    for lab in range(labels.max() + 1):
        idx = np.where(labels == lab)[0]
        tc = collections.Counter(t for i in idx for t in tags(ms[i]) if not t.startswith("audit"))
        dates = sorted((ms[i].get("created_at") or ms[i].get("t") or "")[:10] for i in idx)
        topics.append({"topic": f"t{lab:03d}", "ids": [ms[i]["id"] for i in idx], "n": len(idx),
                       "chars": int(chars[idx].sum()), "tags": [t for t, _ in tc.most_common(8)],
                       "span": [dates[0], dates[-1]], "centroid": v[idx].mean(0).round(5).tolist()})
    (a.data / "topics.json").write_text(json.dumps(topics))
    cs = sorted(t["chars"] for t in topics)
    print(f"{len(ms)} memories ({dropped} excluded) -> {len(topics)} topics; "
          f"chars min {cs[0]} median {cs[len(cs) // 2]} max {cs[-1]}")
    for t in sorted(topics, key=lambda t: -t["n"])[:12]:
        print(t["topic"], t["n"], t["chars"], t["tags"][:6])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
