"""Stage 1 — Asymmetric corpus assembly.

Produces two disjoint pools:
  - empirical_corpus.json: cs.LG/CV/CL/NE, stat.ML, cs.CR/NI/DC (2023–2025 bias)
  - theory_corpus.json:    math.PR/ST/IT/OC/FA/AP/CO/MG/DG (any era)

No math×math or math.AG/NT/RT (phase A showed only folklore there at scale).

Output files:
  data/empirical_corpus.json  — {arxiv_ids: [...], n: N}
  data/theory_corpus.json     — {arxiv_ids: [...], n: N}

Usage:
  python te_corpus.py [--target-emp N] [--target-th N] [--seed K]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import save_json  # noqa: E402

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
ARXIV_INTERVAL = 3.5  # seconds between requests

# Empirical pool: ML/applied-CS, biased toward 2023–2025 observations
EMPIRICAL_CATS: dict[str, int] = {
    "cs.LG":  200,
    "cs.CV":  150,
    "cs.CL":  150,
    "cs.NE":   80,
    "stat.ML": 120,
    "cs.CR":   80,
    "cs.NI":   80,
    "cs.DC":   80,
}

# Theory pool: math that doesn't expire
THEORY_CATS: dict[str, int] = {
    "math.PR":  300,
    "math.ST":  250,
    "math.IT":  200,
    "math.OC":  200,
    "math.FA":  200,
    "math.AP":  200,
    "math.CO":  200,
    "math.MG":  100,
    "math.DG":  100,
}

# Recent start offsets for empirical papers (2023+ = low offset)
EMPIRICAL_START_RANGE = (0, 2000)
# Theory has longer history — sample from wider window
THEORY_START_RANGE = (0, 10000)


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
            ids: list[str] = []
            for e in root.findall("a:entry", NS):
                id_el = e.find("a:id", NS)
                if id_el is None or not id_el.text:
                    continue
                raw = id_el.text.rsplit("/abs/", 1)[-1]
                ids.append(raw.split("v")[0])
            return ids
        except (httpx.HTTPError, ET.ParseError) as e:
            wait = 2 ** attempt + 1
            print(f"  arxiv {category} retry {attempt+1}: {e}", file=sys.stderr)
            time.sleep(wait)
    return []


def sample_category(
    category: str,
    n: int,
    *,
    rng: random.Random,
    start_range: tuple[int, int],
) -> list[str]:
    page_size = min(200, max(n * 2, 50))
    start = rng.randint(*start_range)
    ids = arxiv_query(category, start=start, max_results=page_size)
    rng.shuffle(ids)
    return ids[:n]


def build_pool(
    cats: dict[str, int],
    target: int,
    *,
    rng: random.Random,
    start_range: tuple[int, int],
    label: str,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for cat, budget in cats.items():
        time.sleep(ARXIV_INTERVAL)
        sampled = sample_category(cat, budget, rng=rng, start_range=start_range)
        added = 0
        for aid in sampled:
            if aid not in seen:
                seen.add(aid)
                ids.append(aid)
                added += 1
        print(f"  [{label}] {cat}: requested {budget}, +{added} new (total {len(ids)})")
        if len(ids) >= target * 1.2:
            break

    rest = ids[:]
    rng.shuffle(rest)
    return rest[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-emp", type=int, default=800)
    ap.add_argument("--target-th",  type=int, default=1500)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"Assembling empirical pool (target={args.target_emp})…")
    emp_ids = build_pool(
        EMPIRICAL_CATS,
        args.target_emp,
        rng=rng,
        start_range=EMPIRICAL_START_RANGE,
        label="emp",
    )
    save_json("empirical_corpus.json", {"arxiv_ids": emp_ids, "n": len(emp_ids)})

    print(f"\nAssembling theory pool (target={args.target_th})…")
    # Theory pool must not overlap with empirical IDs
    emp_set = set(emp_ids)
    th_ids_raw = build_pool(
        THEORY_CATS,
        args.target_th + 200,  # overshoot to absorb dedup loss
        rng=rng,
        start_range=THEORY_START_RANGE,
        label="th",
    )
    th_ids = [x for x in th_ids_raw if x not in emp_set][: args.target_th]
    save_json("theory_corpus.json", {"arxiv_ids": th_ids, "n": len(th_ids)})

    print(f"\nDone: empirical={len(emp_ids)}, theory={len(th_ids)}")


if __name__ == "__main__":
    main()
