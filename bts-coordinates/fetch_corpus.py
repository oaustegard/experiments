#!/usr/bin/env python3
"""Build a candidate pool per test case by unioning arXiv keyword searches.

Infra ground truth, probed 2026-09-01 (contradicts the 2026-07-25 record in
claude-workspace PR #180, which killed the prior-art-probe on this exact point):
  - arXiv export API keyword search: WORKS. The P2-shaped query returns
    arXiv:2412.05182 (the P2 target) as hit #1.
  - S2 /paper/search (keyword): still dead, 429 on first call, anonymous.
  - S2 /paper/{id} and /references: work, but 429 under sustained sequential use.
  - Doerr's /references is still elided by the publisher (data: null).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "muninn-raven/1.0 (research; oskar@austegard.com)"}
ARXIV = "https://export.arxiv.org/api/query"
MIN_INTERVAL = 3.5
_last = [0.0]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def arxiv_search(query, max_results=40, retries=3):
    url = f"{ARXIV}?search_query={urllib.parse.quote('all:' + query)}&max_results={max_results}"
    for attempt in range(retries):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return _parse(r.read().decode()), None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(4 * (attempt + 1))
    return [], err


def _parse(xml):
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        def g(tag):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else None
        aid = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
        if not aid:
            continue
        out.append({
            "id": "arXiv:" + aid.group(1).split("v")[0],
            "arxiv_id": aid.group(1),
            "title": g("title"),
            "abstract": g("summary"),
            "published": g("published"),
        })
    return out


def build_pool(queries, cutoff="2026-07-21", max_results=100, log=None):
    """Union arXiv results over the query list, dedupe, date-filter."""
    pool, per_query = {}, {}
    for q in queries:
        hits, err = arxiv_search(q, max_results=max_results)
        kept = 0
        for h in hits:
            if h["published"] and h["published"][:10] >= cutoff:
                continue
            if h["id"] not in pool:
                pool[h["id"]] = h
                pool[h["id"]]["found_by"] = [q]
            else:
                pool[h["id"]]["found_by"].append(q)
            kept += 1
        per_query[q] = {"returned": len(hits), "kept": kept, "error": err}
        msg = f"  {len(hits):3d} hits, {kept:3d} kept  <- {q}" + (f"   ERR {err}" if err else "")
        print(msg, flush=True)
        if log is not None:
            log.append(msg)
    return pool, per_query


DOERR = {
    "id": "DOI:10.1007/s00493-004-0007-x",
    "arxiv_id": None,
    "title": "Linear Discrepancy of Totally Unimodular Matrices",
    # S2 returns abstract: null - elided by the publisher. Recorded, not invented.
    "abstract": None,
    "published": "2004-01-01T00:00:00Z",
    "found_by": ["INJECTED - not on arXiv, see PLAN.md confound 2"],
}


def main():
    case = sys.argv[1]
    qfile = sys.argv[2]
    queries = json.load(open(qfile))["queries"]
    print(f"[{case}] {len(queries)} queries", flush=True)
    pool, per_query = build_pool(queries)
    if case.startswith("P1"):
        pool[DOERR["id"]] = dict(DOERR)
    out = {"case": case, "queries": queries, "per_query": per_query,
           "pool": list(pool.values()), "n": len(pool)}
    path = os.path.join(CACHE, f"pool_{case}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"[{case}] pool n={len(pool)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
