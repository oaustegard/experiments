"""Tier 1 — Targeted 4810-paper corpus assembly + SPECTER2 fetch.

Subfield budgets are biased toward known-bridge-fertile pairs (math.CO,
math.PR, cs.CC, cs.LG, cs.IT) per the tier-0 findings. Total: 4800
sampled + 10 phase-0 anchors.

Writes to $PHASE_A_DATA_DIR (default: mvp/phase_a/tier1).
"""
from __future__ import annotations

import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, chunked, load_json, s2_post, save_json  # noqa: E402
from remax import RemaxBuilder  # noqa: E402
from stage1_corpus import ANCHOR_IDS, NS  # noqa: E402

CATEGORY_BUDGET: dict[str, int] = {
    "math.NT": 200,
    "math.AG": 200,
    "math.CO": 250,
    "math.RT": 150,
    "math.PR": 200,
    "cs.CC": 250,
    "cs.IT": 200,
    "cs.LG": 250,
    "cs.LO": 150,
}

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_INTERVAL = 8.0   # arXiv heavily 429s back-to-back queries
ARXIV_PAGE = 200
BATCH_SIZE = 500
SPECTER_DIM = 768
REMAX_K = 2
INTERBATCH_SLEEP = 4.0
SEED = 7  # tier-1 seed; distinct from runs 1/2

EXTRA_VECTORS_PATH = Path("/tmp/sawin_lenstra.json")
if not EXTRA_VECTORS_PATH.exists():
    # Try repo-root version (PR #89)
    alt = Path(__file__).resolve().parents[3] / "sawin_lenstra_specter.json"
    if alt.exists():
        EXTRA_VECTORS_PATH = alt
EXTRA_LABEL_TO_ARXIV = {
    "sawin": "2605.20579",
    "openai_companion": "2605.20695",
}


def arxiv_query(category: str, *, start: int, max_results: int) -> list[str]:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(5):
        try:
            r = httpx.get(ARXIV_API, params=params, timeout=60, follow_redirects=True)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ids = []
            for e in root.findall("a:entry", NS):
                id_el = e.find("a:id", NS)
                if id_el is None or not id_el.text:
                    continue
                raw = id_el.text.rsplit("/abs/", 1)[-1]
                aid = raw.split("v")[0]
                ids.append(aid)
            return ids
        except (httpx.HTTPError, ET.ParseError) as e:
            wait = 2 ** attempt + 1
            print(f"  arxiv {category} retry {attempt + 1}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return []


def sample_category(category: str, n: int, *, rng: random.Random) -> list[str]:
    """Pull n papers, paginating if needed. With n=700 and page=500 we issue
    two arXiv calls per category."""
    out: list[str] = []
    start = rng.randint(0, 5000)
    while len(out) < n:
        page = arxiv_query(category, start=start, max_results=min(ARXIV_PAGE, n - len(out) + 50))
        if not page:
            break
        out.extend(page)
        start += ARXIV_PAGE
        time.sleep(ARXIV_INTERVAL)
    rng.shuffle(out)
    return out[:n]


def assemble_corpus() -> list[str]:
    rng = random.Random(SEED)
    ids: list[str] = list(ANCHOR_IDS)
    seen: set[str] = set(ids)
    for cat, budget in CATEGORY_BUDGET.items():
        time.sleep(ARXIV_INTERVAL)
        sampled = sample_category(cat, budget, rng=rng)
        added = 0
        for aid in sampled:
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
                added += 1
        print(f"  {cat}: +{added}/{len(sampled)}  (corpus={len(ids)})")
    return ids


def s2_batch_with_embeddings(arxiv_ids: list[str]) -> list[dict]:
    payload = {"ids": [f"ARXIV:{a}" for a in arxiv_ids]}
    fields = "paperId,title,externalIds,fieldsOfStudy,s2FieldsOfStudy,embedding.specter_v2"
    return s2_post("/paper/batch", payload, params={"fields": fields})


def fetch_specter(ids: list[str]) -> tuple[RemaxBuilder, list[dict], list[str]]:
    builder = RemaxBuilder(d=SPECTER_DIM, k=REMAX_K, n_estimated=len(ids))
    all_meta: list[dict] = []
    missing: list[str] = []
    for batch_idx, batch in enumerate(chunked(ids, BATCH_SIZE)):
        print(f"  S2 batch {batch_idx}: {len(batch)} ids…")
        t0 = time.time()
        resp = s2_batch_with_embeddings(batch)
        vecs: list[list[float]] = []
        pids: list[str] = []
        for arxiv_id, paper in zip(batch, resp):
            if not paper:
                missing.append(arxiv_id)
                continue
            emb = (paper.get("embedding") or {}).get("vector")
            if not emb:
                missing.append(arxiv_id)
                continue
            pid = paper.get("paperId") or arxiv_id
            vecs.append(emb)
            pids.append(pid)
            all_meta.append({
                "paperId": pid,
                "arxiv_id": (paper.get("externalIds") or {}).get("ArXiv") or arxiv_id,
                "title": paper.get("title") or "",
                "fos": paper.get("fieldsOfStudy") or [],
                "s2_fos": [f.get("category") for f in (paper.get("s2FieldsOfStudy") or [])],
            })
        if vecs:
            v = np.asarray(vecs, dtype=np.float32)
            builder.append(pids, builder.project_and_pack(v))
        print(f"    fetched {len(vecs)} in {time.time() - t0:.1f}s ({len(batch) - len(vecs)} missing)")
        time.sleep(INTERBATCH_SLEEP)
    return builder, all_meta, missing


def inject_external_specter(
    builder: RemaxBuilder, all_meta: list[dict], missing: list[str], corpus_ids: list[str]
) -> tuple[list[dict], list[str]]:
    if not EXTRA_VECTORS_PATH.exists():
        return all_meta, missing
    import json as _json
    extra = _json.loads(EXTRA_VECTORS_PATH.read_text())
    injected: list[str] = []
    vecs: list[list[float]] = []
    for label, arxiv_id in EXTRA_LABEL_TO_ARXIV.items():
        e = extra.get(label)
        if not e:
            continue
        if arxiv_id not in corpus_ids:
            continue
        if arxiv_id not in missing:
            continue
        vecs.append(e["vector"])
        injected.append(arxiv_id)
        all_meta.append({
            "paperId": f"ext:{arxiv_id}",
            "arxiv_id": arxiv_id,
            "title": e.get("title") or "",
            "fos": ["Mathematics"],
            "s2_fos": [],
            "source": "phase0_external_specter2",
        })
    if vecs:
        v = np.asarray(vecs, dtype=np.float32)
        builder.append([f"ext:{a}" for a in injected], builder.project_and_pack(v))
        print(f"  injected {len(injected)} external SPECTER2 vectors: {injected}")
        missing = [m for m in missing if m not in injected]
    return all_meta, missing


def main() -> None:
    print(f"Tier 1: assembling targeted corpus -> {DATA}")

    # Stage 1: corpus.
    include = load_json("include_ids.json")
    if include:
        ids = include["all"]
        print(f"  reusing existing include_ids.json ({len(ids)} ids)")
    else:
        ids = assemble_corpus()
        save_json("include_ids.json", {"anchors": ANCHOR_IDS, "all": ids, "n": len(ids), "seed": SEED})
    print(f"\nCorpus: {len(ids)} arXiv IDs")

    # Stage 2: SPECTER2.
    if (DATA / "codes.bin").exists() and load_json("metadata.json"):
        print("  SPECTER2 codes already on disk; skipping fetch.")
        return
    builder, all_meta, missing = fetch_specter(ids)
    all_meta, missing = inject_external_specter(builder, all_meta, missing, ids)

    codes = builder.partial_codes()
    (DATA / "codes.bin").write_bytes(codes.tobytes())
    save_json("metadata.json", all_meta)
    save_json("missing.json", {"n": len(missing), "ids": missing})
    save_json("codes_meta.json", {
        "n": codes.shape[0], "bytes_per_vec": codes.shape[1],
        "bits_per_vec": codes.shape[1] * 8, "d": SPECTER_DIM, "k": REMAX_K, "seed": builder.seed,
    })
    print(f"\nDone. codes.bin = {codes.shape}, missing {len(missing)}")


if __name__ == "__main__":
    main()
