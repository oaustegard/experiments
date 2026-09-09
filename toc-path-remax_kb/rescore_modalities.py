"""Re-score the per-modality legs with chunk ids resolved from row numbers.

`KB._dense_search` and `KB._bm25_search` return Hit objects whose `chunk_id` is
still the empty string — only `KB.search` fills it in. The first run compared
that empty string against gold ids and scored a clean 0.000 for every arm and
every k, which is a failed measurement rather than a null result.

The fused ranks from the first run are unaffected and are carried over
untouched; only the dense and bm25 legs are recomputed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/remax_kb")
sys.path.insert(0, str(Path(__file__).parent))

from remax_kb.read_v2 import KB
from run_arms import (ARMS, DATA, HERE, BatchedJinaQ4, b2_gold_map, load_chunks,
                      rank_of, score)


class CachedEmbedder(BatchedJinaQ4):
    """Every arm scores the same queries with the same embedder, so a query's
    vector is identical across arms. Cache it: 4x fewer forward passes."""

    _cache: dict[tuple[str, str], np.ndarray] = {}

    def encode(self, texts, *, prompt):
        if len(texts) == 1:
            key = (texts[0], prompt)
            hit = CachedEmbedder._cache.get(key)
            if hit is not None:
                return hit
            v = super().encode(texts, prompt=prompt)
            CachedEmbedder._cache[key] = v
            return v
        return super().encode(texts, prompt=prompt)


def ids_from_rows(kb: KB, hits, limit: int = 50) -> list[str]:
    return [kb._chunk_id_at(h.row) for h in hits[:limit]]


def main() -> None:
    gold = [json.loads(l) for l in (DATA / "gold_chunks.jsonl").read_text().splitlines()]
    qrecs = [json.loads(l) for l in (DATA / "queries.jsonl").read_text().splitlines()]
    b2map = b2_gold_map(gold, load_chunks("B2"))

    queries = [r["query"] for r in qrecs]
    gold_a = [{r["gold_id"]} for r in qrecs]
    gold_b2 = [b2map.get(r["gold_id"], set()) for r in qrecs]

    gold_by_id = {g["id"]: g for g in gold}
    ctrl_ids = sorted({r["gold_id"] for r in qrecs})
    ctrl_pairs = [(gold_by_id[i]["heading_path"], {i}) for i in ctrl_ids
                  if gold_by_id[i]["heading_path"]]
    ctrl_q = [p[0] for p in ctrl_pairs]
    ctrl_gold = [p[1] for p in ctrl_pairs]
    ctrl_gold_b2 = [b2map.get(next(iter(p[1])), set()) for p in ctrl_pairs]

    embedder = CachedEmbedder()
    results = json.loads((HERE / "results.json").read_text())

    for arm in ARMS:
        kb = KB.open(HERE / "kbs" / arm / f"{arm}.kbi")
        ga = gold_b2 if arm == "B2" else gold_a
        gc = ctrl_gold_b2 if arm == "B2" else ctrl_gold

        for tag, qs, gs, rk, sk in (
            ("main", queries, ga, "ranks", "arms"),
            ("ctrl", ctrl_q, gs_ := gc, "control_ranks", "control"),
        ):
            dense, bm25 = [], []
            for q, g in zip(qs, gs if tag == "main" else gs_):
                dense.append(rank_of(ids_from_rows(kb, kb._dense_search(q, embedder)), g))
                bm25.append(rank_of(ids_from_rows(kb, kb._bm25_search(q)), g))
            results[rk][arm]["dense"] = dense
            results[rk][arm]["bm25"] = bm25
            results[sk][arm]["dense"] = score(dense)
            results[sk][arm]["bm25"] = score(bm25)

        a = results["arms"][arm]
        print(f"{arm}: dense R@1 {a['dense']['R@1']:.3f}  bm25 R@1 {a['bm25']['R@1']:.3f}  "
              f"fused R@1 {a['fused']['R@1']:.3f}", flush=True)

    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(f"rewrote results.json ({len(CachedEmbedder._cache)} cached query embeds)")


if __name__ == "__main__":
    main()
