"""Stage 1 — Corpus assembly.

Build a 1000-paper include_ids.json:
  - 10 phase-0 anchor papers (must be included)
  - random sample from arXiv math.* + cs-theory categories, biased toward
    bridge-relevant content

We use arXiv's API (free, lightly rate-limited) to enumerate candidates by
category, then S2 batch to filter to those that have SPECTER2 vectors.
The issue spec suggested S2 /recommendations + /search but those need an
API key for any real throughput; arXiv works without one.

Output: data/include_ids.json with arXiv IDs (raw form, e.g. "2103.09508").
"""
from __future__ import annotations

import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json  # noqa: E402

# ---------------------------------------------------------------------------
# Anchors — phase-0 papers that must appear in the corpus.
# Lenstra 1986 is excluded (no arXiv source).
# ---------------------------------------------------------------------------
ANCHORS = [
    ("2103.09508", "HMR — algebraic NT"),
    ("1904.07062", "HMR-v2"),
    ("2507.15679", "Pach-Raz — combinatorial geometry"),
    ("2412.11914", "AMP — combinatorial geometry"),
    ("2002.00502", "Erdős distance 2D"),
    ("2605.20579", "Sawin — realized Erdős bridge"),
    ("2605.20695", "OpenAI companion"),
    ("1402.0290",  "Tao — Navier-Stokes (random-math control)"),
    ("1105.6164",  "Tal-Vardy polar codes (Lenstra-shape candidate)"),
    ("2103.00020", "CLIP (ML deep control)"),
]
ANCHOR_IDS = [a[0] for a in ANCHORS]

# Per-category sample budget. Issue suggests math.* + cs.{LG,AI,CC,DM,IT,FL,LO,DS,GT,CG,IR}.
# Keep it simple: rough quota that totals ~1100 candidates, dedupe down to 1000.
CATEGORY_BUDGET: dict[str, int] = {
    # math
    "math.NT": 120,    # number theory (the algebraic side of the Erdős bridge)
    "math.CO": 120,    # combinatorics (the geometric side)
    "math.AG": 80,     # algebraic geometry (Lenstra-shape bridges)
    "math.PR": 60,
    "math.AP": 50,
    "math.GT": 50,     # geometric topology
    "math.LO": 40,
    "math.RT": 40,
    # cs-theory
    "cs.LG": 80,
    "cs.AI": 50,
    "cs.CC": 60,
    "cs.DM": 80,
    "cs.IT": 80,
    "cs.FL": 30,
    "cs.LO": 30,
    "cs.DS": 60,
    "cs.GT": 30,
    "cs.CG": 50,
    "cs.IR": 30,
}

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

# arXiv asks for ~3s between requests; be polite.
ARXIV_INTERVAL = 3.5


def arxiv_query(category: str, *, start: int, max_results: int) -> list[str]:
    """Fetch a page of arXiv IDs from a category, sorted by submitted date desc."""
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
            ids: list[str] = []
            for e in root.findall("a:entry", NS):
                id_el = e.find("a:id", NS)
                if id_el is None or not id_el.text:
                    continue
                raw = id_el.text.rsplit("/abs/", 1)[-1]
                # Strip version suffix: "2103.09508v3" -> "2103.09508"
                aid = raw.split("v")[0]
                ids.append(aid)
            return ids
        except (httpx.HTTPError, ET.ParseError) as e:
            wait = 2 ** attempt + 1
            print(f"  arxiv {category} retry {attempt + 1}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    return []


def sample_category(category: str, n: int, *, rng: random.Random) -> list[str]:
    """Sample ~n random arXiv IDs from a category.

    Strategy: arXiv returns recent papers in submitted-date order; pull a
    larger page from a randomly-offset window so we don't always get the
    same most-recent papers across runs.
    """
    page_size = min(200, max(n * 2, 50))
    # Random offset within the last ~10k papers in this category so we get
    # variety across runs without paginating deep.
    start = rng.randint(0, 8000)
    ids = arxiv_query(category, start=start, max_results=page_size)
    rng.shuffle(ids)
    return ids[:n]


def build_include_ids(target: int = 1000, *, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    ids: list[str] = list(ANCHOR_IDS)
    seen: set[str] = set(ids)

    # Sample each category. arXiv rate-limit means ~20 categories * 3.5s = ~70s.
    for cat, budget in CATEGORY_BUDGET.items():
        time.sleep(ARXIV_INTERVAL)
        sampled = sample_category(cat, budget, rng=rng)
        added = 0
        for aid in sampled:
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
                added += 1
        print(f"  {cat}: requested {budget}, got {len(sampled)}, +{added} new (total {len(ids)})")
        if len(ids) >= target * 1.1:
            break

    # Trim/pad to target.
    # Anchors always in front; sample of the rest.
    anchor_set = set(ANCHOR_IDS)
    rest = [i for i in ids if i not in anchor_set]
    rng.shuffle(rest)
    final = ANCHOR_IDS + rest[: target - len(ANCHOR_IDS)]
    return final


def main() -> None:
    import os as _os
    seed = int(_os.environ.get("PHASE_A_SEED", "42"))
    target = int(_os.environ.get("PHASE_A_TARGET", "1000"))
    print(f"Phase A stage 1: assembling {target}-paper include_ids (seed={seed})…")
    ids = build_include_ids(target=target, seed=seed)
    print(f"\nTotal: {len(ids)} unique IDs ({len(ANCHOR_IDS)} anchors + {len(ids) - len(ANCHOR_IDS)} sampled)")
    save_json("include_ids.json", {
        "anchors": ANCHOR_IDS,
        "all": ids,
        "n": len(ids),
    })


if __name__ == "__main__":
    main()
