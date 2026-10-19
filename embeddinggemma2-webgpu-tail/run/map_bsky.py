"""2D and 3D maps of the Bluesky corpus for web/map.html: UMAP layouts, topics, labels, time.

Usage: python3 -I run/map_bsky.py [--topics 32] [--neighbors 15] [--min-dist 0.02]

Input: out/bsky/emb.npy and out/bsky/posts.jsonl (run/build_bsky_index.py's order, which is
web/bsky/posts.json's order). Steps:
  1. PCA to 50 dims of the unit embeddings (keeps cosine geometry, makes UMAP fast).
  2. UMAP to 2D and, separately, to 3D (cosine, seed 42; 15 neighbours and min_dist 0.02 by default,
     which separates topics more than 30 / 0.05 did on this corpus).
  3. k-means topics on the 50-dim PCA; each topic labelled by its top c-TF-IDF words
     (a word's frequency in the topic over its frequency in the corpus).
  4. Monthly post counts per topic, for the share-over-time view.
Writes web/bsky/map.u8.txt (float32 xy[n,2], float32 xyz[n,3], uint32 time[n] in seconds,
uint8 topic[n], little-endian, in that order; .txt because the Pages builder does not ship .bin) and web/bsky/map.json. No post text is written;
the labels are aggregate keywords.
"""
import argparse
import collections
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bsky_index import load_posts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out" / "bsky"
WEB = ROOT / "web" / "bsky"

STOP = set("""a about above after again against all also am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll
she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves just like get got one really think know people thing things
going even much way make see said say also still now well yeah yes lot good new time want need go use using used
https http www com bsky app social would could it's i'm don't that's can't they're doesn't didn't isn't there's
re ve ll amp via lol oh ok okay maybe actually pretty right since many every something someone anything back first
will org great today day exist
""".split())
WORD = re.compile(r"[a-z][a-z0-9'\-]{2,}")


def words(text):
    text = re.sub(r"https?://\S+|@\S+", " ", text.lower())
    return [w.strip("'-") for w in WORD.findall(text) if w.strip("'-") not in STOP and len(w.strip("'-")) > 2]


def label_topics(texts, topic, k, top=4):
    df_topic = [collections.Counter() for _ in range(k)]
    df_all = collections.Counter()
    for t, c in zip(texts, topic):
        ws = set(words(t))
        df_topic[c].update(ws)
        df_all.update(ws)
    n = len(texts)
    sizes = np.bincount(topic, minlength=k)
    labels = []
    for c in range(k):
        scored = []
        for w, f in df_topic[c].items():
            if f < max(5, 0.01 * sizes[c]):
                continue
            # share of the topic's posts using w, over the share of all posts using it
            scored.append((f / sizes[c] * math.log(n / df_all[w]), w))
        scored.sort(reverse=True)
        labels.append([w for _, w in scored[:top]])
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=32)
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.02)
    a = ap.parse_args()
    rows, authors = load_posts()
    X = np.load(SRC / "emb.npy").astype(np.float32)
    assert len(X) == len(rows), (len(X), len(rows))
    t0 = time.time()
    P = PCA(50, random_state=42).fit_transform(X)
    print(f"PCA {time.time() - t0:.1f}s", flush=True)
    layouts = {}
    for dims in (2, 3):
        t0 = time.time()
        Y = umap.UMAP(n_components=dims, n_neighbors=a.neighbors, min_dist=a.min_dist, metric="cosine",
                      random_state=42, low_memory=True).fit_transform(P)
        Y = (Y - Y.mean(0)) / np.abs(Y - Y.mean(0)).max()       # centred, in [-1, 1]
        layouts[dims] = Y.astype(np.float32)
        print(f"UMAP {dims}D {time.time() - t0:.1f}s", flush=True)
    topic = KMeans(a.topics, n_init=4, random_state=42).fit_predict(P).astype(np.uint8)
    labels = label_topics([r["text"] for r in rows], topic, a.topics)

    ts = np.array([int(datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).timestamp())
                   if r.get("created_at") else 0 for r in rows], np.int64)
    ts = np.clip(ts, 0, 2**32 - 1).astype(np.uint32)
    months = sorted({datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m") for t in ts if t})
    midx = {m: i for i, m in enumerate(months)}
    monthly = np.zeros((a.topics, len(months)), np.int64)
    for t, c in zip(ts, topic):
        if t:
            monthly[c, midx[datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m")]] += 1

    topics = []
    for c in range(a.topics):
        m = topic == c
        topics.append({"id": c, "label": " · ".join(labels[c][:3]), "words": labels[c], "size": int(m.sum()),
                       "c2": [round(float(v), 4) for v in np.median(layouts[2][m], 0)],
                       "c3": [round(float(v), 4) for v in np.median(layouts[3][m], 0)],
                       "monthly": monthly[c].tolist()})
    WEB.mkdir(parents=True, exist_ok=True)
    with open(WEB / "map.u8.txt", "wb") as f:
        f.write(layouts[2].astype("<f4").tobytes())
        f.write(layouts[3].astype("<f4").tobytes())
        f.write(ts.astype("<u4").tobytes())
        f.write(topic.tobytes())
    valid = ts[ts > 0]
    (WEB / "map.json").write_text(json.dumps({
        "n": len(rows), "topics": topics, "months": months, "month_totals": monthly.sum(0).tolist(),
        "time_min": int(valid.min()), "time_max": int(valid.max()),
        "layout": {"pca": 50, "umap": {"n_neighbors": a.neighbors, "min_dist": a.min_dist, "metric": "cosine", "seed": 42},
                   "kmeans_k": a.topics},
        "binary": "float32 xy[n*2], float32 xyz[n*3], uint32 time_s[n], uint8 topic[n], little-endian",
        "order": "same rows as posts.json",
    }, separators=(",", ":")))
    for t in sorted(topics, key=lambda t: -t["size"]):
        print(f"{t['size']:6d}  {t['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
