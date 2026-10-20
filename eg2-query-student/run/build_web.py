"""Inputs for web/embed.html, the page that computes EG2 targets on Oskar's GPU.

Usage: python3 -I run/build_web.py

Writes web/items.json: the texts to embed, in shard order. A string is a query
(out/queries/synth_more.jsonl); an integer is a post index whose snippet the page builds
itself from text it fetches from Bluesky, so no post text is published. Posts are every
post of >= 40 characters outside the 1,500 test-pool posts, the same pool synth_more drew
from. Writes web/parity.json: 256 synth_train queries with their CPU EG2 vectors, which the
page embeds first and compares (the run is only usable if they agree).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_synth  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SHARD = 30000


def snippet(text):
    """The post text a snippet target is computed from. web/embed.js has the same rule:
    whitespace runs to one space, then the first 200 code points, cut back to the last
    space when that leaves at least 120."""
    t = " ".join(text.split())
    if len(t) <= 200:
        return t
    cut = t[:200]
    sp = cut.rfind(" ")
    return cut[:sp] if sp >= 120 else cut


def write_parity():
    qs = [json.loads(l)["q"] for l in (ROOT / "out" / "queries" / "synth_train.jsonl").open()][:256]
    E = np.load(ROOT / "out" / "queries" / "synth_train.eg2.npy")[:256]
    (WEB / "parity.json").write_text(json.dumps(
        {"queries": qs, "vectors": [[round(float(x), 5) for x in v] for v in E]}))


def main():
    WEB.mkdir(exist_ok=True)
    write_parity()
    more = [json.loads(l)["q"] for l in (ROOT / "out" / "queries" / "synth_more.jsonl").open()]
    rows = gen_synth.posts()
    random.Random(0).shuffle(rows)
    test = {r["i"] for r in rows[gen_synth.N_TRAIN:gen_synth.N_TRAIN + gen_synth.N_TEST]}
    post_ids = sorted(r["i"] for r in rows if r["i"] not in test)
    items = more + post_ids
    (WEB / "items.json").write_text(json.dumps({"shard": SHARD, "items": items}, ensure_ascii=False))
    print(json.dumps({"queries": len(more), "posts": len(post_ids), "items": len(items),
                      "shards": -(-len(items) // SHARD)}))


if __name__ == "__main__":
    main()
