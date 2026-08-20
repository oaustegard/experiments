#!/usr/bin/env python3
"""Let each arm run at the granularity it is good at, and pick the chunk that answers.

`RESULTS.md` ran BM25 and the dense arm at one shared granularity, which turned
out to suit only one of them. Moving from 31,169 example-level chunks to the
6,397 pages they came from gained BM25 **+0.061** gold-in-sources and moved the
dense arms +0.012, −0.054 and +0.006. That is an asymmetry with a mechanism: a
longer document gives BM25 more term coverage and its length normalization
absorbs the cost, while a mean-pooled vector over a whole page averages away the
one example that matched.

So this measures the cross product rather than the diagonal — BM25 over pages
for *which utility*, the encoder over chunks for *which example* — and the two
stages it enables:

**Coarse (which utility).** Every combination of BM25 granularity x dense
granularity, fused at utility level. Fusing after aggregation is what makes the
mixed arm expressible at all: a page and a chunk are not the same object and
cannot be fused as documents, but "the score this arm gives utility `u`" is the
same object either way.

**Fine (which text).** The generator gets `tldr[u][0]` today — the utility's
first tldr example, ordered by however the page was written, with no reference
to the request. Once the coarse stage has named the utility, choosing among its
examples is free: they are already encoded. `sources_form` picks between the
first example, the example the query scores highest, and the whole page.

Only the fine stage can move end-to-end routing without moving retrieval, because
`gold_in_sources` is a utility-level metric and is blind to which example was
sent. That is also why the fine stage needs `fullsystem_dense.py --source-form`
to be scored at all; this script measures the coarse grid and writes the chunk
choice that the generator run then consumes.

    python3 coarse_to_fine.py --models leaf-mt-int8 minilm-l6-int8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
SELFHIST = experiment("nl2sh-selfhist")
sys.path.insert(0, str(RETRIEVAL))
import retrieve as R  # noqa: E402
import dense_index as D  # noqa: E402
from eval_dense import load_eval, mcnemar, tldr_from_chunks  # noqa: E402


class Side:
    """One arm at one granularity, exposing a utility ranking for any query."""

    def __init__(self, granularity: str, chunks_path: Path, model: str | None,
                 adapter: Path | None, pool: int) -> None:
        self.granularity = granularity
        self.model = model
        self.pool = pool
        chunks = D.GRANULARITIES[granularity](R.load_chunks(chunks_path))
        self.utilities = np.array([c.utility for c in chunks], dtype=object)
        if model is None:
            self.index = R.Index(chunks)
            self.dense = None
        else:
            _, _, self.dense = D.load(model, chunks_path, granularity=granularity,
                                      adapter=adapter)
            self.index = None

    def rank(self, nl: str) -> list[tuple[str, float]]:
        if self.dense is None:
            return D.rank_utilities(self.index.scores(nl), self.utilities,
                                    self.pool, positive_only=True)
        return D.rank_utilities(self.dense.scores(nl), self.utilities, self.pool)


def sources_from_ranking(ranking: list[tuple[str, float]], tldr: dict,
                         k: int) -> list[str]:
    """The k utilities the generator would be shown, from a utility ranking.

    `eval_dense` derived this from a pool of top chunks, which a mixed-granularity
    arm has no single version of. Taking it from the fused utility ranking is the
    same quantity by a route that works for every arm, so every number in this
    file is internally comparable — and it is re-derived here for the published
    same-granularity arms so the comparison is like for like.
    """
    out = []
    for u, _ in ranking:
        if u in tldr:
            out.append(u)
        if len(out) >= k:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["leaf-mt-int8"])
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--nl", type=Path, nargs="+",
                    default=[SELFHIST / "cyber_nl.json", HERE / "cyber_nl_ext.json"])
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--tldr", type=Path, default=None)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7])
    ap.add_argument("--out", type=Path, default=HERE / "results_coarse_to_fine.json")
    a = ap.parse_args()

    base_chunks = R.load_chunks(a.chunks)
    if a.tldr:
        import pleias_gate as G
        tldr = G.load_tldr(a.tldr)
    else:
        tldr = tldr_from_chunks(base_chunks)

    rows: list[dict] = []
    seen = set()
    for path in a.nl:
        for r in load_eval(path, tldr)[0]:
            if r["nl"] not in seen:
                seen.add(r["nl"])
                rows.append(r)
    clean = [r for r in rows if not r.get("names_utility")]
    print(f"eval: {len(rows)} rows, {len(clean)} leak-free", file=sys.stderr)

    bm = {g: Side(g, a.chunks, None, None, a.pool) for g in ("chunk", "page")}
    results = {"config": {"k": a.k, "pool": a.pool, "n": len(rows),
                          "n_leak_free": len(clean),
                          "adapter": str(a.adapter) if a.adapter else None},
               "arms": []}

    def record(name: str, ranker) -> dict:
        per = []
        for r in rows:
            srcs = sources_from_ranking(ranker(r["nl"]), tldr, a.k)
            per.append({"nl": r["nl"], "utility": r["utility"],
                        "names_utility": bool(r.get("names_utility")),
                        "gold_at_top1": bool(srcs) and srcs[0] == r["utility"],
                        "gold_in_sources": r["utility"] in srcs,
                        "sources": srcs})
        ok = [p for p in per if not p["names_utility"]]
        arm = {"arm": name,
               "gold_at_top1": round(sum(p["gold_at_top1"] for p in ok) / len(ok), 3),
               "gold_in_sources": round(sum(p["gold_in_sources"] for p in ok) / len(ok), 3),
               "rows": per}
        results["arms"].append(arm)
        return arm

    for g in ("chunk", "page"):
        record(f"bm25@{g}", bm[g].rank)

    for model in a.models:
        dn = {g: Side(g, a.chunks, model, a.adapter, a.pool) for g in ("chunk", "page")}
        for g in ("chunk", "page"):
            record(f"dense@{g}:{model}", dn[g].rank)
        for bg in ("chunk", "page"):
            for dg in ("chunk", "page"):
                def rrf_rank(nl, bg=bg, dg=dg):
                    return D.rrf(bm[bg].rank(nl), dn[dg].rank(nl))
                record(f"rrf:bm25@{bg}+{model}@{dg}", rrf_rank)
                for alpha in a.alphas:
                    def w_rank(nl, bg=bg, dg=dg, alpha=alpha):
                        return D.wsum(bm[bg].rank(nl), dn[dg].rank(nl), alpha)
                    record(f"wsum{alpha}:bm25@{bg}+{model}@{dg}", w_rank)

    base = results["arms"][0]
    for arm in results["arms"][1:]:
        arm["vs_bm25_chunk"] = {
            m: mcnemar([p[m] for p in base["rows"] if not p["names_utility"]],
                       [p[m] for p in arm["rows"] if not p["names_utility"]])
            for m in ("gold_at_top1", "gold_in_sources")}

    a.out.write_text(json.dumps(results, indent=1) + "\n")
    hdr = f"{'arm':<40}{'gold@1':>8}{'sources':>9}{'p':>9}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for arm in results["arms"]:
        p = arm.get("vs_bm25_chunk", {}).get("gold_in_sources", {}).get("p")
        print(f"{arm['arm']:<40}{arm['gold_at_top1']:>8.3f}{arm['gold_in_sources']:>9.3f}"
              f"{('' if p is None else f'{p:.4f}'):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
