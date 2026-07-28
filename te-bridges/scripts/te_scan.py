"""Stage 3 — Asymmetric cross-axis band scan + hard dedup.

Computes cosine similarity between empirical SPECTER2 vectors and theory
SPECTER2 vectors (no empirical×empirical or theory×theory pairs).

Hard dedup tiers (in order):
  1. Sequential arXiv prefix (same YYMM + numeric IDs within 50): cheap proxy
     for same-author group.
  2. Author-set Jaccard overlap: block any pair sharing any author.
  3. Direct citation overlap: drop pairs where one cites the other (requires
     S2 references field; only used if S2_API_KEY is set and --use-citations
     flag is passed, since it adds significant S2 call cost).

Output:
  data/te_candidates.json  — top-N pairs after dedup, sorted by cosine sim

Usage:
  python te_scan.py [--top-k 2000] [--use-citations]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import S2_API_KEY, load_json, s2_get, s2_post, save_json  # noqa: E402

DEFAULT_TOP_K = 2000


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def is_sequential_pair(a: str, b: str) -> bool:
    if "." not in a or "." not in b:
        return False
    pa, na = a.split(".", 1)
    pb, nb = b.split(".", 1)
    if pa != pb:
        return False
    try:
        return abs(int(na.split("v")[0]) - int(nb.split("v")[0])) <= 50
    except ValueError:
        return False


def build_author_map(meta_list: list[dict]) -> dict[str, set[str]]:
    """Return {arxiv_id -> set of author names} by fetching S2 authors field.

    Batched in groups of 500. Returns empty dict if S2 key unavailable
    (author dedup is best-effort; not blocking).
    """
    if not S2_API_KEY:
        print("  author dedup: S2_API_KEY not set, skipping", file=sys.stderr)
        return {}

    result: dict[str, set[str]] = {}
    ids = [m["arxiv_id"] for m in meta_list if m.get("arxiv_id")]
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        try:
            resp = s2_post(
                "/paper/batch",
                {"ids": [f"ARXIV:{a}" for a in chunk]},
                params={"fields": "externalIds,authors"},
            )
            for arxiv_id, paper in zip(chunk, resp):
                if not paper:
                    continue
                authors = {
                    a.get("name", "").strip().lower()
                    for a in (paper.get("authors") or [])
                    if a.get("name")
                }
                result[arxiv_id] = authors
            time.sleep(2.0)
        except Exception as e:
            print(f"  author fetch batch error: {e}", file=sys.stderr)
    return result


def fetch_citations(arxiv_ids: list[str]) -> dict[str, set[str]]:
    """Return {arxiv_id -> set of cited arxiv_ids} via S2 /paper/batch references.

    Expensive: one S2 call per paper (references not batchable cleanly).
    Only called when --use-citations is set.
    """
    result: dict[str, set[str]] = {}
    for arxiv_id in arxiv_ids:
        try:
            data = s2_get(
                f"/paper/ARXIV:{arxiv_id}/references",
                params={"fields": "externalIds", "limit": 200},
            )
            cited: set[str] = set()
            for ref in data.get("data") or []:
                ext = (ref.get("citedPaper") or {}).get("externalIds") or {}
                aid = ext.get("ArXiv")
                if aid:
                    cited.add(aid)
            result[arxiv_id] = cited
            time.sleep(0.5)
        except Exception as e:
            print(f"  cite fetch {arxiv_id}: {e}", file=sys.stderr)
            result[arxiv_id] = set()
    return result


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--use-citations", action="store_true")
    args = ap.parse_args()

    emp_meta = load_json("empirical_meta.json")
    th_meta  = load_json("theory_meta.json")
    if not emp_meta or not th_meta:
        print("ERROR: run te_embed.py first", file=sys.stderr)
        sys.exit(1)

    emp_vecs = np.load(str(Path(__file__).resolve().parent.parent / "data" / "empirical_vecs.npy"))
    th_vecs  = np.load(str(Path(__file__).resolve().parent.parent / "data" / "theory_vecs.npy"))
    print(f"Loaded: empirical={emp_vecs.shape}, theory={th_vecs.shape}")

    # Normalize for cosine
    emp_norm = emp_vecs / (np.linalg.norm(emp_vecs, axis=1, keepdims=True) + 1e-9)
    th_norm  = th_vecs  / (np.linalg.norm(th_vecs,  axis=1, keepdims=True) + 1e-9)

    print("Computing cross-axis cosine similarity matrix…")
    # Shape: (N_emp, N_th) — memory estimate at 800×1500: ~4.8M floats = ~19MB
    S = emp_norm @ th_norm.T
    print(f"  sim matrix shape: {S.shape}, range [{S.min():.3f}, {S.max():.3f}]")

    emp_arxiv = [m["arxiv_id"] for m in emp_meta]
    th_arxiv  = [m["arxiv_id"] for m in th_meta]
    emp_title = {m["arxiv_id"]: m.get("title","") for m in emp_meta}
    th_title  = {m["arxiv_id"]: m.get("title","") for m in th_meta}

    # Flatten top-K by cosine sim (no band; we want continuous ranking)
    print(f"Extracting top {args.top_k} pairs by cosine similarity…")
    flat = S.ravel()
    if len(flat) <= args.top_k:
        top_flat_idx = np.argsort(flat)[::-1]
    else:
        # argpartition + argsort for speed
        part_idx = np.argpartition(flat, -args.top_k)[-args.top_k:]
        top_flat_idx = part_idx[np.argsort(flat[part_idx])[::-1]]

    n_emp, n_th = S.shape
    pairs_raw: list[dict] = []
    for idx in top_flat_idx:
        i, j = divmod(int(idx), n_th)
        pairs_raw.append({
            "cosine_sim": float(S[i, j]),
            "emp_arxiv": emp_arxiv[i],
            "th_arxiv": th_arxiv[j],
            "emp_title": emp_title.get(emp_arxiv[i], ""),
            "th_title":  th_title.get(th_arxiv[j], ""),
        })
    print(f"  raw top-{len(pairs_raw)} pairs extracted")

    # Dedup tier 1: sequential arXiv prefix
    pre = len(pairs_raw)
    pairs_raw = [
        p for p in pairs_raw
        if not is_sequential_pair(p["emp_arxiv"], p["th_arxiv"])
    ]
    print(f"  sequential dedup: dropped {pre - len(pairs_raw)}, {len(pairs_raw)} remain")

    # Dedup tier 2: author-set Jaccard
    all_arxiv = list({p["emp_arxiv"] for p in pairs_raw} | {p["th_arxiv"] for p in pairs_raw})
    author_map = build_author_map(emp_meta + th_meta)
    if author_map:
        pre = len(pairs_raw)
        pairs_raw = [
            p for p in pairs_raw
            if not (
                author_map.get(p["emp_arxiv"], set()) &
                author_map.get(p["th_arxiv"], set())
            )
        ]
        print(f"  author-Jaccard dedup: dropped {pre - len(pairs_raw)}, {len(pairs_raw)} remain")

    # Dedup tier 3: citation overlap (optional, expensive)
    if args.use_citations and S2_API_KEY:
        print("  fetching citations for dedup (this takes a while)…")
        cites = fetch_citations(all_arxiv)
        pre = len(pairs_raw)
        pairs_raw = [
            p for p in pairs_raw
            if p["th_arxiv"] not in cites.get(p["emp_arxiv"], set())
            and p["emp_arxiv"] not in cites.get(p["th_arxiv"], set())
        ]
        print(f"  citation dedup: dropped {pre - len(pairs_raw)}, {len(pairs_raw)} remain")

    out = {
        "n_pairs": len(pairs_raw),
        "dedup_tiers": ["sequential_arxiv", "author_jaccard"]
        + (["citation"] if args.use_citations and S2_API_KEY else []),
        "pairs": pairs_raw,
    }
    save_json("te_candidates.json", out)
    print(f"\nDone: {len(pairs_raw)} candidate pairs saved to data/te_candidates.json")


if __name__ == "__main__":
    main()
