#!/usr/bin/env python3
"""Does a dense arm lift retrieval on the independent cyber eval?

Issue #48's finding: the on-device system is retrieval-bound. The fine-tuned
Gemma routes 0.706 when the gold utility's page is in its context and 0.206 when
real BM25 picks the context, because BM25 surfaces the gold utility only 26% of
the time. So this measures the retrieval tier alone, on the same eval, before
any generator runs.

Two metrics, both reproducing an existing baseline number so the arms are
comparable to what is already written down:

* **utility-level** (`calibrate.py`'s view) — every utility ranked by its best
  chunk. BM25 baseline: gold@1 0.118, gold@k=3 0.235, gold@show=5 0.324, n=34.
* **sources** (`gemma_fullsystem.py`'s view) — the top-15 *chunks*, reduced to
  the first 3 distinct utilities that have a tldr example. This is literally
  what goes into the generator's prompt. BM25 baseline: 0.263 over n=38.

Arms: BM25 alone, dense alone, RRF, and a min-max weighted sum over an alpha
sweep. Both fusions are run because `hybrid-code-index` won 24/24 with RRF while
`gh-mcp-regex-fit` measured RRF losing to a weighted sum when the arms were of
unequal quality — and BM25 at 0.118 gold@1 is the unequal case.

    python3 eval_dense.py --models leaf-mt-int8 bekko-a8m
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def tldr_from_chunks(chunks: list) -> dict[str, list[tuple[str, str]]]:
    """Rebuild `pleias_gate.load_tldr`'s dict from the committed chunk corpus.

    The corpus was built from the tldr pages, one chunk per example, text laid
    out as "description\\ncommand" — so the pages directory is not needed to
    reconstruct it, and the eval runs from a bare checkout. `--tldr` still
    accepts the real directory; `--check-tldr` asserts the two agree.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for c in chunks:
        if c.kind != "tldr_example":
            continue
        desc, _, cmd = c.text.partition("\n")
        if cmd:
            out.setdefault(c.utility, []).append((desc, cmd))
    return out


def load_eval(nl_path: Path, tldr: dict) -> tuple[list, list]:
    rows = [r for r in json.loads(nl_path.read_text())
            if r.get("nl") and r["utility"] in tldr]
    clean = [r for r in rows if not r.get("names_utility")]
    return rows, clean


def sources_utilities(ranked_chunks: list[int], chunks: list, tldr: dict,
                      k: int) -> list[str]:
    """`gemma_fullsystem.retrieved_sources`, without building the prompt strings."""
    seen: list[str] = []
    for i in ranked_chunks:
        u = chunks[i].utility
        if u in seen or u not in tldr:
            continue
        seen.append(u)
        if len(seen) >= k:
            break
    return seen


def top_chunks(scores: np.ndarray, n: int, positive_only: bool) -> list[int]:
    k = min(n, len(scores))
    cand = np.argpartition(-scores, k - 1)[:k]
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    return [int(i) for i in cand if not positive_only or scores[i] > 0]


def summarize(per: list[dict], k: int, show: int) -> dict:
    n = len(per)
    clean = [p for p in per if not p["names_utility"]]
    def rate(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None
    return {
        "n": n, "n_leak_free": len(clean),
        "gold_at_top1": rate(clean, "gold_at_top1"),
        f"gold_in_top{k}": rate(clean, "gold_in_topk"),
        f"gold_in_show{show}": rate(clean, "gold_in_show"),
        "gold_in_sources_all": rate(per, "gold_in_sources"),
        "gold_in_sources_leak_free": rate(clean, "gold_in_sources"),
        "mean_rank_of_gold": round(float(np.mean([p["gold_rank"] for p in clean
                                                  if p["gold_rank"] is not None])), 1)
        if any(p["gold_rank"] is not None for p in clean) else None,
        "gold_ranked_at_all": rate(clean, "gold_ranked"),
    }


def run_arm(name: str, rankers, rows: list, chunks: list, tldr: dict,
            k: int, show: int) -> dict:
    """`rankers` maps a row to (utility_ranking, chunk_ranking)."""
    per = []
    t0 = time.time()
    for r in rows:
        util_rank, chunk_rank = rankers(r)
        utils = [u for u, _ in util_rank]
        gold = r["utility"]
        pos = utils.index(gold) + 1 if gold in utils else None
        per.append({
            "utility": gold, "nl": r["nl"],
            "names_utility": bool(r.get("names_utility")),
            "gold_at_top1": bool(utils) and utils[0] == gold,
            "gold_in_topk": gold in utils[:k],
            "gold_in_show": gold in utils[:show],
            "gold_in_sources": gold in sources_utilities(chunk_rank, chunks, tldr, k),
            "gold_rank": pos, "gold_ranked": pos is not None,
        })
    out = summarize(per, k, show)
    out["arm"] = name
    out["wall_s"] = round(time.time() - t0, 1)
    out["rows"] = per
    return out


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact paired test on two binary outcome vectors over the same queries.

    n is 34-165 here, so a two-proportion test on the aggregates would ignore
    that both arms answered the same queries and would be badly under-powered.
    Only the discordant pairs carry information: b01 queries the first arm won,
    b10 the second. Two-sided exact binomial p under H0 that a discordant pair is
    a coin flip.
    """
    from math import comb

    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"wins_a": 0, "wins_b": 0, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins_a": b01, "wins_b": b10, "p": round(p, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["leaf-mt-int8"])
    ap.add_argument("--nl", type=Path, nargs="+", default=[SELFHIST / "cyber_nl.json"],
                    help="one or more cyber_nl-shaped files; rows are concatenated")
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--tldr", type=Path, default=None,
                    help="tldr pages dir; default reconstructs it from the chunks")
    ap.add_argument("--granularity", default="chunk", choices=["chunk", "page"])
    ap.add_argument("--reform", nargs="*", default=[], choices=["rm3", "dense-prf"])
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--pool", type=int, default=400, help="chunk pool for utility ranking")
    ap.add_argument("--sources-pool", type=int, default=15,
                    help="chunk pool the generator's source list is drawn from")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--no-utility-prefix", action="store_true")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="query-side linear adapter from adapter.py")
    ap.add_argument("--out", type=Path, default=HERE / "results_dense.json")
    a = ap.parse_args()

    chunks = D.GRANULARITIES[a.granularity](R.load_chunks(a.chunks))
    index = R.Index(chunks)
    utilities = np.array([c.utility for c in chunks], dtype=object)
    tldr = tldr_from_chunks(chunks) if not a.tldr else None
    if a.tldr:
        sys.path.insert(0, str(RETRIEVAL))
        import pleias_gate as G
        tldr = G.load_tldr(a.tldr)

    rows: list[dict] = []
    for path in a.nl:
        rows.extend(load_eval(path, tldr)[0])
    seen, uniq = set(), []
    for r in rows:                       # the extended sample excludes the
        if r["nl"] in seen:              # original commands, but a duplicate
            continue                     # request would double-count a query
        seen.add(r["nl"])
        uniq.append(r)
    rows = uniq
    clean = [r for r in rows if not r.get("names_utility")]
    print(f"eval: {len(rows)} rows, {len(clean)} leak-free, "
          f"{len(set(r['utility'] for r in clean))} distinct gold utilities",
          file=sys.stderr)

    results = {"config": {"k": a.k, "show": a.show, "pool": a.pool,
                          "sources_pool": a.sources_pool, "n": len(rows),
                          "n_leak_free": len(clean),
                          "granularity": a.granularity,
                          "documents": len(chunks),
                          "nl": [str(p.name) for p in a.nl],
                          "adapter": str(a.adapter) if a.adapter else None,
                          "utility_prefix": not a.no_utility_prefix},
               "arms": []}

    bm_cache = {r["nl"]: index.scores(r["nl"]) for r in rows}

    def bm_rankers(r):
        s = bm_cache[r["nl"]]
        return (D.rank_utilities(s, utilities, a.pool, positive_only=True),
                top_chunks(s, a.sources_pool, positive_only=True))

    results["arms"].append(run_arm("bm25", bm_rankers, rows, chunks, tldr, a.k, a.show))

    for model in a.models:
        _, _, dense = D.load(model, a.chunks, not a.no_utility_prefix,
                             granularity=a.granularity, adapter=a.adapter)
        dn_cache = {r["nl"]: dense.scores(r["nl"]) for r in rows}
        size_mb = round(dense.enc.artifact_bytes() / 1e6, 1)

        def add(name, fn):
            arm = run_arm(name, fn, rows, chunks, tldr, a.k, a.show)
            arm["artifact_mb"] = size_mb
            results["arms"].append(arm)

        tag = model + ("+adapter" if a.adapter else "")
        add(f"dense:{tag}", lambda r: (
            D.rank_utilities(dn_cache[r["nl"]], utilities, a.pool),
            top_chunks(dn_cache[r["nl"]], a.sources_pool, positive_only=False)))

        def fused_chunks(bs, ds, n):
            """Chunk-level RRF for the sources metric, same k=60 as the utility fusion."""
            acc: dict[int, float] = {}
            for ranking in (top_chunks(bs, 200, positive_only=True),
                            top_chunks(ds, 200, positive_only=False)):
                for rank, i in enumerate(ranking, start=1):
                    acc[i] = acc.get(i, 0.0) + 1.0 / (60 + rank)
            return [i for i, _ in sorted(acc.items(), key=lambda x: -x[1])[:n]]

        def wsum_chunks(bs, ds, alpha, n):
            def norm(v):
                lo, hi = float(v.min()), float(v.max())
                return (v - lo) / max(hi - lo, 1e-9)
            return top_chunks((1 - alpha) * norm(bs) + alpha * norm(ds), n,
                              positive_only=False)

        def rrf_rankers(r):
            bs, ds = bm_cache[r["nl"]], dn_cache[r["nl"]]
            fused = D.rrf(D.rank_utilities(bs, utilities, a.pool, positive_only=True),
                          D.rank_utilities(ds, utilities, a.pool))
            return fused, fused_chunks(bs, ds, a.sources_pool)

        add(f"rrf:bm25+{tag}", rrf_rankers)

        for alpha in a.alphas:
            def w_rankers(r, alpha=alpha):
                bs, ds = bm_cache[r["nl"]], dn_cache[r["nl"]]
                fused = D.wsum(
                    D.rank_utilities(bs, utilities, a.pool, positive_only=True),
                    D.rank_utilities(ds, utilities, a.pool), alpha)
                return fused, wsum_chunks(bs, ds, alpha, a.sources_pool)

            add(f"wsum{alpha}:bm25+{tag}", w_rankers)

        if a.reform:
            import reformulate as RF
            for how in a.reform:
                rewritten = {r["nl"]: (RF.rm3(index, r["nl"]) if how == "rm3"
                                       else RF.dense_prf(index, dense, r["nl"]))
                             for r in rows}
                rf_cache = {q: index.scores(rewritten[q]) for q in rewritten}

                def rf_rankers(r, c=rf_cache):
                    s = c[r["nl"]]
                    return (D.rank_utilities(s, utilities, a.pool, positive_only=True),
                            top_chunks(s, a.sources_pool, positive_only=True))

                add(f"bm25+{how}" + (f"({model})" if how == "dense-prf" else ""),
                    rf_rankers)

    base = results["arms"][0]
    for arm in results["arms"][1:]:
        arm["vs_bm25"] = {
            metric: mcnemar([p[metric] for p in base["rows"] if not p["names_utility"]],
                            [p[metric] for p in arm["rows"] if not p["names_utility"]])
            for metric in ("gold_at_top1", "gold_in_topk", "gold_in_sources")}

    a.out.write_text(json.dumps(results, indent=1) + "\n")

    hdr = (f"{'arm':<30}{'gold@1':>8}{'gold@' + str(a.k):>8}{'gold@' + str(a.show):>8}"
           f"{'sources':>9}{'p(src)':>9}{'MB':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for arm in results["arms"]:
        p = arm.get("vs_bm25", {}).get("gold_in_sources", {}).get("p")
        print(f"{arm['arm']:<30}{arm['gold_at_top1']:>8.3f}"
              f"{arm['gold_in_top' + str(a.k)]:>8.3f}"
              f"{arm['gold_in_show' + str(a.show)]:>8.3f}"
              f"{arm['gold_in_sources_leak_free']:>9.3f}"
              f"{('' if p is None else f'{p:.3f}'):>9}"
              f"{arm.get('artifact_mb', 0):>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
