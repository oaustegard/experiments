"""Stage 0 (anchor-mode) — Targeted candidate assembly via S2 search + LLM-crafted queries.

Replaces te_corpus + te_embed + te_scan when running in anchor-seeded mode.
Reads anchors.json, for each anchor:
  1. Fetches anchor metadata (title, abstract, paperId, SPECTER2) from S2
  2. Generates 5-7 opposite-pool query reformulations via Gemini
  3. Runs S2 bulk-search per reformulation (filtered to opposite-pool fields + year range, arxiv-only)
  4. Also pulls S2 recommendations as a small bonus arm
  5. Unions + dedups by arxiv_id, fetches SPECTER2 for the union
  6. Cosine-ranks against the anchor, keeps top-K
  7. Emits te_candidates.json in the same schema downstream stages expect:
     pairs of (emp_arxiv, th_arxiv) where the anchor sits on its declared side.

Output (in $TE_DATA_DIR or anchor_run/data/):
  anchor_meta.json       — {arxiv_id: {paperId, title, abstract, vec}}
  anchor_reformulations.json — {arxiv_id: [query strings]}
  anchor_candidates_raw.json — {arxiv_id: [{paperId, arxiv, title, year, source}]}
  anchor_neighbor_specter.json — {paperId: vec | null} for all union candidates
  te_candidates.json     — schema-compatible with downstream stages 4-7

Usage:
  TE_DATA_DIR=mvp/theory-empirical-bridges/anchor_run/data python3 te_anchor.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import (  # noqa: E402
    DATA,
    S2_API_KEY,
    ascii_fold,
    chunked,
    gemini_generate,
    load_json,
    retry,
    s2_get,
    s2_headers,
    s2_post,
    save_json,
)

# -------- knobs ------------------------------------------------------------
RECS_LIMIT       = 30        # S2 recommendations per anchor
SEARCH_LIMIT     = 50        # S2 bulk-search results per reformulation
RECS_PAUSE       = 1.5       # politeness pause between S2 calls (no API key)
SEARCH_PAUSE     = 1.5
TOP_K_PER_ANCHOR = 80        # final neighbors kept per anchor after SPECTER2 cosine

# Map opposite-pool intent to S2 fieldsOfStudy filters
POOL_FIELDS_S2 = {
    "empirical": ["Computer Science"],
    "theory":    ["Mathematics", "Computer Science"],  # math + theoretical CS
}

# Map opposite-pool intent to a downstream arXiv-category whitelist
# (used to label candidates and to enforce a sanity filter on the union)
POOL_ARXIV_CATS = {
    "empirical": {
        "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML",
        "cs.CR", "cs.NI", "cs.DC", "cs.AI",
    },
    "theory": {
        "math.PR", "math.ST", "math.IT", "math.OC", "math.FA",
        "math.AP", "math.CO", "math.MG", "math.DG", "math.NA",
        "cs.CC", "cs.DM", "cs.LO", "cs.IT",
    },
}


REFORMULATION_PROMPT = """You are helping assemble a candidate-paper search to find {direction} papers semantically related to this {anchor_pool} paper:

Title: {title}

Abstract:
{abstract}

The candidate pool we want to search is {direction} papers. Your job is to produce 6 short keyword-style search queries (3-8 words each) that, when run against a research-paper search index, would return papers WHOSE OBSERVATIONS THIS THEOREM EXPLAINS / WHOSE THEOREMS THIS OBSERVATION INVOKES (depending on direction).

Constraints:
- Each query must be in the language of the opposite pool. For an empirical/ML target, use ML/CS terminology (e.g. "neural network expressivity", "symbolic regression neural"). For a theoretical-math target, use math terminology (e.g. "operator basis function decomposition", "universal approximation Stone-Weierstrass").
- Each query should target a DIFFERENT facet of the connection — different concrete phenomenon or different mathematical object. Do not just paraphrase the title.
- Avoid generic queries that would return millions of results ("deep learning", "neural network"). Each query should be specific enough to return a few hundred to a few thousand candidates.
- Output JSON only: {{"queries": ["query 1", "query 2", ...]}} with exactly 6 strings.
"""


def fetch_anchor_meta(arxiv_id: str) -> dict | None:
    """Get paperId, title, abstract, and SPECTER2 for a single anchor."""
    fields = "paperId,title,abstract,externalIds,embedding.specter_v2"
    try:
        r = s2_get(f"/paper/ARXIV:{arxiv_id}", params={"fields": fields})
    except Exception as e:
        print(f"  anchor fetch {arxiv_id}: {e}", file=sys.stderr)
        return None
    if not r:
        return None
    emb = (r.get("embedding") or {}).get("vector")
    return {
        "arxiv_id": arxiv_id,
        "paperId": r.get("paperId"),
        "title": r.get("title", ""),
        "abstract": r.get("abstract", "") or "",
        "vec": emb,
    }


def llm_reformulate(anchor: dict, opposite_pool: str) -> list[str]:
    """Ask Gemini to produce 6 opposite-pool search queries for this anchor."""
    direction = {
        "empirical": "applied-ML / empirical-CS",
        "theory":    "theoretical-mathematics",
    }[opposite_pool]
    anchor_pool = "theoretical-math" if opposite_pool == "empirical" else "applied-ML"
    abstract = anchor.get("abstract", "") or "(no abstract available)"
    prompt = REFORMULATION_PROMPT.format(
        direction=direction,
        anchor_pool=anchor_pool,
        title=anchor["title"],
        abstract=abstract[:2000],
    )
    raw = gemini_generate(prompt, model="gemini-2.5-flash",
                          json_mode=True, max_tokens=512, thinking_budget=0)
    s = raw.strip().lstrip("`").lstrip("json").lstrip()
    s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        print(f"  llm reformulate parse fail; raw={raw[:200]}", file=sys.stderr)
        return []
    try:
        obj = json.loads(s[start:end+1])
        qs = obj.get("queries") or []
        return [q for q in qs if isinstance(q, str) and 3 <= len(q.split()) <= 12]
    except json.JSONDecodeError:
        return []


def s2_recommendations(paper_id: str) -> list[dict]:
    """Pull up to RECS_LIMIT recommendations (no S2 key required for low rate).

    Recommendations live at a different base path than /graph/v1, so we call
    httpx directly here rather than via te_common.s2_get.
    """
    url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
    params = {"limit": RECS_LIMIT,
              "fields": "paperId,title,externalIds,fieldsOfStudy,year,authors"}

    def _call():
        with httpx.Client(timeout=60.0) as c:
            r = c.get(url, params=params, headers=s2_headers())
            if r.status_code == 429:
                raise httpx.HTTPError(f"recs 429: {r.text[:200]}")
            r.raise_for_status()
            return r.json()

    try:
        data = retry(_call, attempts=6, cap=30.0)
    except Exception as e:
        print(f"  recs fetch {paper_id}: {e}", file=sys.stderr)
        return []
    return data.get("recommendedPapers") or []


def s2_bulk_search(query: str, opposite_pool: str, year_range: str) -> list[dict]:
    """Bulk-search papers matching `query`, filtered to opposite-pool S2 fieldsOfStudy."""
    fields = "paperId,title,externalIds,fieldsOfStudy,year,authors"
    foS = ",".join(POOL_FIELDS_S2.get(opposite_pool, []))
    try:
        r = s2_get("/paper/search/bulk", params={
            "query": query,
            "year": year_range,
            "fieldsOfStudy": foS,
            "fields": fields,
            "limit": SEARCH_LIMIT,
        })
    except Exception as e:
        print(f"  bulk-search '{query}' err: {e}", file=sys.stderr)
        return []
    return (r.get("data") or []) if r else []


# ---------------------------------------------------------------------------
# Citation-graph exclusions: papers the anchor cites or is cited by
# ---------------------------------------------------------------------------

CITATION_PAGE_LIMIT = 100      # S2 default page size
CITATION_MAX_PAGES  = 10       # cap to 1000 entries per direction
CITATION_PAUSE      = 1.5      # politeness between paginated calls


def _harvest_citation_pages(path: str, key: str) -> tuple[set[str], set[str], set[str]]:
    """Walk paginated /citations or /references for one anchor.
    Returns (arxiv_ids, paperIds, author_names_lowered) drawn from `key` ('citingPaper' or 'citedPaper').
    """
    arx_set: set[str] = set()
    pid_set: set[str] = set()
    author_set: set[str] = set()
    offset = 0
    for _ in range(CITATION_MAX_PAGES):
        try:
            r = s2_get(path, params={
                "fields": "externalIds,authors,paperId",
                "limit":  CITATION_PAGE_LIMIT,
                "offset": offset,
            })
        except Exception as e:
            print(f"  citation page err at {path} offset={offset}: {e}", file=sys.stderr)
            break
        rows = (r.get("data") or []) if isinstance(r, dict) else []
        if not rows:
            break
        for row in rows:
            sub = row.get(key) or {}
            if not sub:
                continue
            pid = sub.get("paperId")
            arx = (sub.get("externalIds") or {}).get("ArXiv")
            if pid:
                pid_set.add(pid)
            if arx:
                arx_set.add(arx)
            for a in (sub.get("authors") or []):
                nm = (a.get("name") or "").strip().lower()
                if nm:
                    author_set.add(nm)
        if len(rows) < CITATION_PAGE_LIMIT:
            break
        offset += CITATION_PAGE_LIMIT
        time.sleep(CITATION_PAUSE)
    return arx_set, pid_set, author_set


def fetch_anchor_exclusions(anchor: dict, meta: dict) -> dict:
    """Build the citation + co-authorship exclude set for one anchor.

    Includes:
      - arxiv/paperIds in /paper/{id}/citations (papers citing the anchor)
      - arxiv/paperIds in /paper/{id}/references (papers the anchor cites)
      - lowercase author names from anchor's own author list
      - lowercase author names from all papers in citations + references
        (to broaden author-overlap dedup to the anchor's immediate research circle)
    """
    pid = meta.get("paperId")
    if not pid:
        return {"arxiv_ids": [], "paperIds": [], "authors": []}

    print(f"  [{anchor['arxiv_id']}] fetching citation graph (citations + references)…")
    citing_arx, citing_pids, citing_authors = _harvest_citation_pages(
        f"/paper/{pid}/citations", "citingPaper")
    print(f"    citers: {len(citing_pids)} papers, {len(citing_authors)} unique authors")
    time.sleep(CITATION_PAUSE)

    cited_arx, cited_pids, cited_authors = _harvest_citation_pages(
        f"/paper/{pid}/references", "citedPaper")
    print(f"    cited:  {len(cited_pids)} papers, {len(cited_authors)} unique authors")
    time.sleep(CITATION_PAUSE)

    # Anchor's own authors via a quick paper lookup
    own_authors: set[str] = set()
    try:
        r = s2_get(f"/paper/{pid}", params={"fields": "authors"})
        for a in ((r or {}).get("authors") or []):
            nm = (a.get("name") or "").strip().lower()
            if nm:
                own_authors.add(nm)
    except Exception as e:
        print(f"    own-authors fetch err: {e}", file=sys.stderr)

    return {
        "arxiv_ids": sorted(citing_arx | cited_arx),
        "paperIds":  sorted(citing_pids | cited_pids),
        "authors":   sorted(own_authors | citing_authors | cited_authors),
        "stats": {
            "n_citing": len(citing_pids),
            "n_cited":  len(cited_pids),
            "n_authors": len(own_authors | citing_authors | cited_authors),
        },
    }


def keep_paper(p: dict, opposite_pool: str) -> bool:
    """Filter: must have arxiv id AND have at least one opposite-pool-relevant arxiv category."""
    aid = (p.get("externalIds") or {}).get("ArXiv")
    if not aid:
        return False
    # Sanity filter on arXiv category if we can derive it. S2 doesn't return primary
    # arXiv category directly, so we accept any arxiv-having paper at this stage.
    # The pipeline's slot extractor + judge handle pool-membership rigor downstream.
    return True


def candidate_arxiv_id(p: dict) -> str | None:
    aid = (p.get("externalIds") or {}).get("ArXiv")
    return aid


def fetch_specter_batch(paper_ids: list[str]) -> dict[str, list[float] | None]:
    """Batch-fetch SPECTER2 for a list of S2 paperIds. Returns {paperId: vec|None}."""
    out: dict[str, list[float] | None] = {}
    fields = "paperId,embedding.specter_v2"
    for chunk in chunked(paper_ids, 500):
        try:
            resp = s2_post("/paper/batch", {"ids": list(chunk)},
                           params={"fields": fields})
        except Exception as e:
            print(f"  specter batch err: {e}", file=sys.stderr)
            resp = []
        for pid, paper in zip(chunk, resp):
            if not paper:
                out[pid] = None
                continue
            emb = (paper.get("embedding") or {}).get("vector")
            out[pid] = emb
        time.sleep(2.0)
    return out


def fetch_abstracts_batch(paper_ids: list[str]) -> dict[str, str]:
    """Batch-fetch abstracts for a list of S2 paperIds. Returns {paperId: abstract}.

    Used by the abstract-mention filter to catch direct-citation cases
    that S2's citation graph hasn't yet ingested (typically a multi-week lag
    for newly-uploaded arXiv papers).
    """
    out: dict[str, str] = {}
    for chunk in chunked(paper_ids, 500):
        try:
            resp = s2_post("/paper/batch", {"ids": list(chunk)},
                           params={"fields": "paperId,abstract"})
        except Exception as e:
            print(f"  abstract batch err: {e}", file=sys.stderr)
            resp = []
        for pid, paper in zip(chunk, resp):
            if not paper:
                out[pid] = ""
                continue
            out[pid] = paper.get("abstract") or ""
        time.sleep(2.0)
    return out


def _anchor_title_keys(title: str) -> list[str]:
    """Generate substrings of the anchor's title to scan in candidate abstracts.
    Title-fragment match is conservative: only catches abstracts that quote
    distinctive multi-word slices of the title. ASCII-folded for unicode safety.
    """
    title = ascii_fold((title or "").strip())
    if not title:
        return []
    words = [w for w in title.split() if len(w) > 2]
    keys = set()
    if len(words) >= 4:
        keys.add(" ".join(words[:5]))    # first 5 distinctive words
        keys.add(" ".join(words[-4:]))   # last 4 words
    if len(title) > 20:
        keys.add(title)                  # the whole thing
    return [k for k in keys if k]


def _anchor_author_keys(authors: list[str]) -> set[str]:
    """Last-name keys for anchor authors. ASCII-folded. Surname-only catches
    citation patterns like 'Cranmer (2023)' or 'Odrzywolek introduced...' in
    candidate abstracts even when S2 normalizes diacritics.
    """
    keys = set()
    for a in (authors or []):
        folded = ascii_fold(a)
        if not folded:
            continue
        last = folded.split()[-1] if folded.split() else ""
        if len(last) > 3:
            keys.add(last)
    return keys


def abstract_mentions_anchor(abstract: str, title_keys: list[str],
                             author_keys: set[str]) -> tuple[bool, str]:
    """Return (drop, reason). Abstract is ASCII-folded before scanning so it
    matches the title_keys / author_keys (which are also ASCII-folded)."""
    if not abstract:
        return False, ""
    a = ascii_fold(abstract)
    for k in title_keys:
        if k in a:
            return True, f"title:'{k[:30]}'"
    for k in author_keys:
        idx = a.find(k)
        if idx >= 0:
            before = a[idx - 1] if idx > 0 else " "
            after = a[idx + len(k)] if idx + len(k) < len(a) else " "
            if not (before.isalnum() or after.isalnum()):
                return True, f"author:'{k}'"
    return False, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="anchors.json",
                    help="anchor config file (in $TE_DATA_DIR)")
    ap.add_argument("--top-k", type=int, default=TOP_K_PER_ANCHOR,
                    help="neighbors to keep per anchor after SPECTER2 cosine")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip Gemini reformulations; use anchor title only")
    ap.add_argument("--skip-citation-filter", action="store_true",
                    help="skip /citations + /references walks (free-tier S2 GETs are heavily rate-limited; "
                         "abstract-mention filter still runs via batch endpoint). For cross-domain anchors "
                         "the citation-graph filter is mostly irrelevant since bridge candidates are by "
                         "definition outside the anchor's citation neighborhood.")
    args = ap.parse_args()

    cfg = load_json(args.anchors)
    if not cfg or not cfg.get("anchors"):
        print(f"ERROR: no anchors found in {DATA / args.anchors}", file=sys.stderr)
        sys.exit(1)
    anchors = cfg["anchors"]
    print(f"Loaded {len(anchors)} anchors from {args.anchors}")

    # -- Step 1: anchor metadata + SPECTER2 ----------------------------------
    # Batch-fetch all uncached anchors via /paper/batch — single POST is far
    # more rate-limit-friendly on free-tier S2 than individual ARXIV: GETs.
    anchor_meta = load_json("anchor_meta.json", default={}) or {}
    need_fetch = [a for a in anchors if a["arxiv_id"] not in anchor_meta]
    if need_fetch:
        ids = [f"ARXIV:{a['arxiv_id']}" for a in need_fetch]
        print(f"  batch-fetching anchor meta for {len(ids)} anchors…")
        try:
            resp = s2_post("/paper/batch", {"ids": ids}, params={
                "fields": "paperId,title,abstract,externalIds,embedding.specter_v2"
            })
        except Exception as e:
            print(f"  ERROR: batch fetch failed: {e}", file=sys.stderr)
            resp = []
        for a, paper in zip(need_fetch, resp):
            aid = a["arxiv_id"]
            if not paper:
                print(f"  ERROR: anchor {aid} not found in batch", file=sys.stderr)
                continue
            emb = (paper.get("embedding") or {}).get("vector")
            if not emb:
                print(f"  ERROR: anchor {aid} missing SPECTER2", file=sys.stderr)
                continue
            anchor_meta[aid] = {
                "arxiv_id": aid,
                "paperId": paper.get("paperId"),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", "") or "",
                "vec": emb,
            }
            print(f"  fetched {aid}: {paper.get('title','')[:60]}")
    for a in anchors:
        if a["arxiv_id"] in anchor_meta:
            continue
        print(f"  anchor {a['arxiv_id']}: still missing")
    save_json("anchor_meta.json", anchor_meta)

    # -- Step 2: LLM reformulations ------------------------------------------
    reformulations = load_json("anchor_reformulations.json", default={}) or {}
    if not args.skip_llm:
        for a in anchors:
            aid = a["arxiv_id"]
            if aid in reformulations:
                print(f"  reformulations {aid}: cached ({len(reformulations[aid])})")
                continue
            meta = anchor_meta.get(aid)
            if not meta:
                continue
            opposite = "empirical" if a["pool"] == "theory" else "theory"
            print(f"  reformulating anchor {aid} → {opposite}…")
            qs = llm_reformulate(meta, opposite)
            reformulations[aid] = qs
            for q in qs:
                print(f"     - {q}")
            time.sleep(1.0)
        save_json("anchor_reformulations.json", reformulations)

    # -- Step 3: S2 recs + bulk search ---------------------------------------
    raw = load_json("anchor_candidates_raw.json", default={}) or {}
    for a in anchors:
        aid = a["arxiv_id"]
        if aid in raw and raw[aid].get("candidates"):
            print(f"  raw candidates {aid}: cached ({len(raw[aid]['candidates'])})")
            continue
        meta = anchor_meta.get(aid)
        if not meta:
            continue
        opposite = "empirical" if a["pool"] == "theory" else "theory"
        year_range = a.get("opposite_year_range", "2018-2026")

        candidates: dict[str, dict] = {}  # arxiv_id -> {paperId, title, ...}

        # Arm A: S2 recommendations
        if meta.get("paperId"):
            print(f"  [{aid}] recommendations…")
            for p in s2_recommendations(meta["paperId"]):
                if not keep_paper(p, opposite):
                    continue
                arx = candidate_arxiv_id(p)
                if not arx or arx == aid:
                    continue
                if arx not in candidates:
                    candidates[arx] = {
                        "paperId": p.get("paperId"),
                        "arxiv_id": arx,
                        "title": p.get("title", ""),
                        "year": p.get("year"),
                        "fos": [f.get("category") if isinstance(f, dict) else f
                                for f in (p.get("fieldsOfStudy") or [])],
                        "authors": [a.get("name", "").strip().lower()
                                    for a in (p.get("authors") or []) if a.get("name")],
                        "source": ["recs"],
                    }
                else:
                    candidates[arx]["source"].append("recs")
            time.sleep(RECS_PAUSE)

        # Arm B: bulk search per reformulation
        for q in reformulations.get(aid, []):
            time.sleep(SEARCH_PAUSE)
            print(f"  [{aid}] bulk-search '{q}'…")
            hits = s2_bulk_search(q, opposite, year_range)
            for p in hits:
                if not keep_paper(p, opposite):
                    continue
                arx = candidate_arxiv_id(p)
                if not arx or arx == aid:
                    continue
                if arx not in candidates:
                    candidates[arx] = {
                        "paperId": p.get("paperId"),
                        "arxiv_id": arx,
                        "title": p.get("title", ""),
                        "year": p.get("year"),
                        "fos": [f.get("category") if isinstance(f, dict) else f
                                for f in (p.get("fieldsOfStudy") or [])],
                        "authors": [a.get("name", "").strip().lower()
                                    for a in (p.get("authors") or []) if a.get("name")],
                        "source": [f"search:{q}"],
                    }
                else:
                    candidates[arx]["source"].append(f"search:{q}")

        raw[aid] = {
            "anchor_arxiv": aid,
            "anchor_pool": a["pool"],
            "opposite_pool": opposite,
            "n_candidates": len(candidates),
            "candidates": list(candidates.values()),
        }
        print(f"  [{aid}] union: {len(candidates)} unique arxiv candidates")
        save_json("anchor_candidates_raw.json", raw)

    # -- Step 3.5: citation-graph + author exclusions ------------------------
    # Defect surfaced in anchor_run/RESULTS: the EML-Sheffer battery paper
    # was a direct citer of Odrzywolek and ranked #2 by SPECTER2 cosine.
    # Hard-exclude: any candidate whose paperId or arxiv_id sits in the
    # anchor's /citations or /references graph, or whose author set
    # overlaps with the anchor's (own authors + collaborators inferred
    # from the citation graph).
    exclusions = load_json("anchor_exclusions.json", default={}) or {}
    if args.skip_citation_filter:
        print("  --skip-citation-filter: skipping /citations + /references walks")
        for a in anchors:
            aid = a["arxiv_id"]
            if aid not in exclusions:
                exclusions[aid] = {"arxiv_ids": [], "paperIds": [], "authors": [],
                                   "stats": {"n_citing": 0, "n_cited": 0, "n_authors": 0,
                                             "skipped": True}}
        save_json("anchor_exclusions.json", exclusions)
    else:
        for a in anchors:
            aid = a["arxiv_id"]
            if aid in exclusions and exclusions[aid].get("stats"):
                ex = exclusions[aid]
                print(f"  exclusions {aid}: cached "
                      f"(n_citing={ex['stats']['n_citing']} n_cited={ex['stats']['n_cited']} "
                      f"n_authors={ex['stats']['n_authors']})")
                continue
            meta = anchor_meta.get(aid)
            if not meta:
                continue
            ex = fetch_anchor_exclusions(a, meta)
            exclusions[aid] = ex
            save_json("anchor_exclusions.json", exclusions)

    # Apply citation + author filter per anchor's candidate list
    pre_total = sum(len(blk["candidates"]) for blk in raw.values())
    for aid, blk in raw.items():
        ex = exclusions.get(aid, {})
        excl_arx = set(ex.get("arxiv_ids", []))
        excl_pids = set(ex.get("paperIds", []))
        excl_authors = set(ex.get("authors", []))
        before = len(blk["candidates"])
        kept = []
        dropped_arxiv = dropped_pid = dropped_author = 0
        for c in blk["candidates"]:
            if c["arxiv_id"] in excl_arx:
                dropped_arxiv += 1
                continue
            if c.get("paperId") and c["paperId"] in excl_pids:
                dropped_pid += 1
                continue
            ca = set(c.get("authors") or [])
            if ca and (ca & excl_authors):
                dropped_author += 1
                continue
            kept.append(c)
        blk["candidates"] = kept
        blk["n_candidates"] = len(kept)
        blk["dedup_dropped"] = {
            "arxiv_match": dropped_arxiv,
            "paperId_match": dropped_pid,
            "author_overlap": dropped_author,
        }
        print(f"  [{aid}] citation+author dedup: {before} → {len(kept)} "
              f"(arxiv:{dropped_arxiv} pid:{dropped_pid} author:{dropped_author})")
    post_total = sum(len(blk["candidates"]) for blk in raw.values())
    print(f"  TOTAL after citation+author dedup: {pre_total} → {post_total}")

    # -- Step 3.6: abstract-mention filter (catches S2 citation-index lag) ----
    # S2 takes weeks-to-months to ingest citations from newly-uploaded arXiv
    # papers. For recent anchors, /citations under-reports — e.g. EML-Sheffer
    # battery paper (May 2026) explicitly cites Odrzywolek (Mar 2026) in its
    # abstract but isn't yet in Odrzywolek's S2 citer list. This filter
    # catches such cases: drop any candidate whose abstract directly
    # mentions the anchor's title fragments or any anchor author surname.
    abstract_cache = load_json("candidate_abstracts.json", default={}) or {}
    need_abstracts = []
    for aid, blk in raw.items():
        for c in blk["candidates"]:
            pid = c.get("paperId")
            if pid and pid not in abstract_cache:
                need_abstracts.append(pid)
    if need_abstracts:
        print(f"\n  fetching abstracts for {len(need_abstracts)} candidates "
              f"(abstract-mention filter)…")
        new_abs = fetch_abstracts_batch(sorted(set(need_abstracts)))
        abstract_cache.update(new_abs)
        save_json("candidate_abstracts.json", abstract_cache)

    pre2_total = sum(len(blk["candidates"]) for blk in raw.values())
    for aid, blk in raw.items():
        meta = anchor_meta.get(aid)
        if not meta:
            continue
        # Build anchor mention keys (title fragments + author surnames)
        anchor_authors = exclusions.get(aid, {}).get("authors", [])
        title_keys = _anchor_title_keys(meta.get("title", ""))
        # Reduce author keys to anchor's OWN surnames only (not the whole
        # citation-graph author pool, which would be too aggressive)
        own_authors = []
        try:
            r = s2_get(f"/paper/{meta['paperId']}", params={"fields": "authors"})
            own_authors = [a.get("name", "") for a in ((r or {}).get("authors") or [])
                           if a.get("name")]
        except Exception:
            pass
        author_keys = _anchor_author_keys(own_authors)
        if not (title_keys or author_keys):
            continue

        kept = []
        dropped_mentions = 0
        drop_examples: list[tuple[str, str]] = []
        for c in blk["candidates"]:
            pid = c.get("paperId")
            abs_text = abstract_cache.get(pid, "")
            drop, reason = abstract_mentions_anchor(abs_text, title_keys, author_keys)
            if drop:
                dropped_mentions += 1
                if len(drop_examples) < 5:
                    drop_examples.append((c["arxiv_id"], reason))
                continue
            kept.append(c)
        blk["candidates"] = kept
        blk["n_candidates"] = len(kept)
        blk.setdefault("dedup_dropped", {})["abstract_mention"] = dropped_mentions
        print(f"  [{aid}] abstract-mention dedup: dropped {dropped_mentions}; "
              f"now {len(kept)} survive")
        for arx, why in drop_examples:
            print(f"     drop {arx}  ({why})")
    post2_total = sum(len(blk["candidates"]) for blk in raw.values())
    print(f"  TOTAL after abstract-mention dedup: {pre2_total} → {post2_total}")
    save_json("anchor_candidates_filtered.json", raw)

    # -- Step 4: SPECTER2 fetch for union (post-filter) ----------------------
    neighbor_spec = load_json("anchor_neighbor_specter.json", default={}) or {}
    all_pids: set[str] = set()
    for aid, blk in raw.items():
        for c in blk.get("candidates", []):
            pid = c.get("paperId")
            if pid and pid not in neighbor_spec:
                all_pids.add(pid)
    if all_pids:
        print(f"\nFetching SPECTER2 for {len(all_pids)} unique candidates…")
        new_specs = fetch_specter_batch(sorted(all_pids))
        neighbor_spec.update(new_specs)
        save_json("anchor_neighbor_specter.json", neighbor_spec)

    # -- Step 5: cosine rank per anchor --------------------------------------
    pairs: list[dict] = []
    for a in anchors:
        aid = a["arxiv_id"]
        meta = anchor_meta.get(aid)
        if not meta or not meta.get("vec"):
            continue
        block = raw.get(aid, {})
        cands = block.get("candidates", [])
        anchor_v = np.asarray(meta["vec"], dtype=np.float32)
        anchor_v = anchor_v / (np.linalg.norm(anchor_v) + 1e-9)
        scored: list[tuple[float, dict]] = []
        for c in cands:
            v = neighbor_spec.get(c.get("paperId"))
            if not v:
                continue
            vv = np.asarray(v, dtype=np.float32)
            vv = vv / (np.linalg.norm(vv) + 1e-9)
            sim = float(anchor_v @ vv)
            scored.append((sim, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        kept = scored[: args.top_k]
        print(f"  [{aid}] {len(scored)} with vectors → kept top {len(kept)}")

        # Emit downstream-compatible pairs
        for sim, c in kept:
            if a["pool"] == "theory":
                emp_arx, emp_title = c["arxiv_id"], c["title"]
                th_arx,  th_title  = aid, meta["title"]
            else:
                emp_arx, emp_title = aid, meta["title"]
                th_arx,  th_title  = c["arxiv_id"], c["title"]
            pairs.append({
                "cosine_sim": round(sim, 6),
                "emp_arxiv": emp_arx,
                "th_arxiv":  th_arx,
                "emp_title": emp_title,
                "th_title":  th_title,
                "source_anchor": aid,
                "source_arm":    c.get("source", []),
            })

    save_json("te_candidates.json", {
        "n_pairs": len(pairs),
        "mode": "anchor",
        "dedup_tiers": [
            "arxiv_id_self_exclude",
            "citation_graph (citing + cited per S2)",
            "author_overlap (anchor + citation-graph authors)",
            "abstract_mention (catches S2 citation-index lag)",
        ],
        "pairs": pairs,
    })

    # Also save anchor_meta and theory/empirical meta files in the schema the
    # downstream stages read, so te_extract can resolve titles per arxiv_id.
    # For anchor mode, both pools are populated from anchor_meta + candidates.
    emp_meta = []
    th_meta  = []
    for a in anchors:
        aid = a["arxiv_id"]
        meta = anchor_meta.get(aid)
        if not meta:
            continue
        entry = {"arxiv_id": aid, "title": meta["title"], "paperId": meta.get("paperId"),
                 "fos": [], "s2_fos": []}
        (th_meta if a["pool"] == "theory" else emp_meta).append(entry)
        opposite = "empirical" if a["pool"] == "theory" else "theory"
        for c in raw.get(aid, {}).get("candidates", []):
            cand_entry = {
                "arxiv_id": c["arxiv_id"], "title": c["title"],
                "paperId": c.get("paperId"), "fos": c.get("fos") or [], "s2_fos": [],
            }
            (emp_meta if opposite == "empirical" else th_meta).append(cand_entry)
    # Dedup by arxiv_id, anchor wins
    def _dedup(lst):
        seen = {}
        for e in lst:
            seen.setdefault(e["arxiv_id"], e)
        return list(seen.values())
    save_json("empirical_meta.json", _dedup(emp_meta))
    save_json("theory_meta.json",    _dedup(th_meta))
    print(f"\nDone: {len(pairs)} candidate pairs in te_candidates.json")


if __name__ == "__main__":
    main()
