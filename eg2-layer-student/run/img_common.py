"""Text->image benchmark on the bsky image-search index (oaustegard.github.io bsky/image-search),
for query encoders in EG2's space. Needs remex 2.1 (out/imgenv): the index is rotation="rht".

Index: the site's committed 2-bit codes (16,878 EG2 q4 image vectors, vision budget 70), scored
as the page does: (R q) . centroids[code] * scale + bias. Queries: alt text of 25-300 characters
(as bsky-image-search/run/eval_query_encoder.py), the query prompt in front, the image as its one
relevant result. Split by image: test = sha1(cid) % 5 == 0 (about 20%), train = the rest, so no
test image's alt text or post text is ever trained on.

Metrics: hit@1 / hit@10 of the source image, and top-10 overlap with full EG2 text's own top 10
over the images. Overlap does not depend on whether the alt text describes the image (alt text
on Bluesky is often commentary), so it is the fidelity metric; hit@k is the task metric.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import remex
from remex.packing import unpack

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "out" / "img"
sys.path.insert(0, str(ROOT.parent / "eg2-query-student" / "run"))
from common import QUERY, boot_ci, overlap  # noqa: E402,F401
from final_eval import paired  # noqa: E402,F401


def is_test(cid):
    return int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 5 == 0


class ImageIndex:
    def __init__(self):
        s = IMG / "site"
        ix = json.loads((s / "index.json").read_text())
        n, d = ix["n"], ix["d"]
        mu = np.array(json.loads(ix["mu"]) if isinstance(ix["mu"], str) else ix["mu"], np.float32)
        self.qz = remex.Quantizer(d, bits=2, seed=ix["seed"], mean=mu, rotation=ix["rotation"])
        packed = np.frombuffer((s / "codes.u8.bin").read_bytes(), np.uint8)
        codes = unpack(packed, 2, n * d).reshape(n, d)
        aux = np.frombuffer((s / "aux.f32.bin").read_bytes(), "<f4").reshape(n, 2)
        self.scale, self.bias = aux[:, 0].copy(), aux[:, 1].copy()
        self.dirs = np.asarray(self.qz.centroids, np.float32)[codes]
        self.n = n

    def scores(self, Q):
        return (np.asarray(self.qz._rotate_rows(np.asarray(Q, np.float32)), np.float32) @ self.dirs.T) * self.scale + self.bias

    def rank_and_top(self, Q, target, k=10, chunk=512):
        rank = np.empty(len(Q), np.int64)
        top = np.empty((len(Q), k), np.int64)
        for s in range(0, len(Q), chunk):
            S = self.scores(Q[s:s + chunk])
            t = target[s:s + chunk]
            rank[s:s + chunk] = (S > S[np.arange(len(t)), t][:, None]).sum(1)
            part = np.argpartition(-S, k, axis=1)[:, :k]
            top[s:s + chunk] = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), 1), 1)
        return rank, top


def rows():
    return [json.loads(l) for l in (IMG / "rows.jsonl").open()]


def eval_queries(split=None):
    """(texts without prompt, target image indices) for alt text 25-300 chars; split in {None,'test','train'}."""
    out = [(r["alt"].strip(), r["i"]) for r in rows()
           if r["alt"] and 25 <= len(r["alt"].strip()) <= 300
           and (split is None or is_test(r["cid"]) == (split == "test"))]
    return [t for t, _ in out], np.array([i for _, i in out])


def train_texts():
    """Caption-domain training texts from train-split images only: every non-empty alt text and
    every distinct non-empty post text (posts can carry several images; a post with any test
    image is excluded entirely)."""
    rs = rows()
    test_posts = {r["uri"] for r in rs if is_test(r["cid"])}
    alts = [r["alt"].strip() for r in rs if r["alt"] and r["alt"].strip() and not is_test(r["cid"])
            and r["uri"] not in test_posts]
    seen, posts = set(), []
    for r in rs:
        t = (r["text"] or "").strip()
        if t and r["uri"] not in test_posts and r["uri"] not in seen:
            seen.add(r["uri"])
            posts.append(t)
    return sorted(set(alts)), posts


def score(I, Q, target, Q_eg2=None, top_eg2=None):
    rank, top = I.rank_and_top(Q, target)
    r = {"hit@1": boot_ci((rank < 1).astype(float)), "hit@10": boot_ci((rank < 10).astype(float))}
    per = {"hit1": (rank < 1).astype(float), "hit10": (rank < 10).astype(float)}
    if top_eg2 is not None:
        ov = overlap(top, top_eg2)
        r["overlap@10_vs_eg2"] = boot_ci(ov)
        r["cos_to_eg2_median"] = float(np.median((Q * Q_eg2).sum(1)))
        per["overlap"] = ov
    return r, per, top
