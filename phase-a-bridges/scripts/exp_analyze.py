"""Experiment analysis.

Answers two questions:

  EXP 2: At full 997-paper Gemini-body scale, where do the known anchor
         pairs rank in pure cosine? Does Gemini-body do as the coarse
         filter what SPECTER2 didn't, **for the easier (Sawin-anchored)
         pairs**? What's the precision of the top-100 — i.e. how many
         anchor pairs make it into the top-100 organically?

  EXP 1: Does gemini-3.5-flash close the gap with Claude subagents on the
         bridge-attempt step? Compare anchor-pair compatibility ratings
         and mediator concepts across the three models.

The user's reframe: precision-on-shortlist matters more than recall. The
right test is "of the organic top-N, how many are real bridges?", not
"does every known bridge surface?".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json  # noqa: E402


ANCHOR_PAIRS = [
    ("2103.09508", "2507.15679", "HMR↔Pach-Raz"),
    ("2605.20579", "2507.15679", "Sawin↔Pach-Raz"),
    ("2605.20579", "2412.11914", "Sawin↔AMP"),
    ("2605.20579", "2103.09508", "Sawin↔HMR"),
    ("2605.20695", "2605.20579", "Sawin↔OpenAI-companion"),
    ("2103.09508", "2412.11914", "HMR↔AMP"),
    ("1105.6164",  "2103.09508", "Tal-Vardy↔HMR"),
    ("2002.00502", "2507.15679", "Erdos2D↔Pach-Raz"),
    ("2002.00502", "2412.11914", "Erdos2D↔AMP"),
]


def exp2_full_corpus_ranking() -> None:
    """Where do anchors rank in full 997-paper Gemini-body cosine space?"""
    print("=" * 70)
    print("EXP 2: Full-corpus Gemini-body selectivity")
    print("=" * 70)
    embs = load_json("full_body_embeddings.json")
    if not embs:
        print("ERROR: full_body_embeddings.json missing; run exp_full_body_embed.py first.")
        return
    embs = {k: v for k, v in embs.items() if v}  # drop nulls
    ids = list(embs.keys())
    X = np.asarray([embs[k] for k in ids], dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = len(ids)
    print(f"Embedded papers: {n}")

    S = X @ X.T
    D = 1 - S
    np.fill_diagonal(D, np.inf)
    iu, ju = np.triu_indices(n, k=1)
    all_d = D[iu, ju]
    sorted_d = np.sort(all_d)
    idx_map = {k: i for i, k in enumerate(ids)}

    print(f"\nAnchor pair ranking in full {n}-paper Gemini-body cosine space:")
    print(f"  ({len(all_d)} total pairs)")
    print()
    anchor_ranks = []
    for a, b, name in ANCHOR_PAIRS:
        if a not in idx_map or b not in idx_map:
            print(f"  {name:30s}  (one paper missing from corpus)")
            continue
        i, j = sorted([idx_map[a], idx_map[b]])
        d = float(D[i, j])
        rank = int((all_d < d).sum())
        pct = 100 * rank / len(all_d)
        anchor_ranks.append((name, d, rank, pct))
        print(f"  {name:30s}  cos_dist={d:.4f}  rank={rank:6d}/{len(all_d)}  (top {pct:.3f}%)")

    # Precision-of-top-N: how many anchor pairs are in the top N?
    print(f"\nPrecision of organic top-N (no force-include):")
    top_ranks = sorted(anchor_ranks, key=lambda x: x[2])
    for N in (20, 50, 100, 200, 500, 1000):
        hit = sum(1 for _, _, r, _ in top_ranks if r < N)
        print(f"  Top-{N:>5d}:  {hit}/{len(top_ranks)} anchor pairs surface organically")

    # Top 20 in full space — what's actually there?
    print(f"\nOrganic top 20 in Gemini-body space (no force-include):")
    order = all_d.argsort()
    for k in range(20):
        p = order[k]
        i, j = iu[p], ju[p]
        is_anchor = any(
            sorted([ids[i], ids[j]]) == sorted([a, b])
            for a, b, _ in ANCHOR_PAIRS
        )
        marker = " ANCHOR" if is_anchor else ""
        print(f"  {k+1:2d}. cos={all_d[p]:.4f}  {ids[i]} ↔ {ids[j]}{marker}")


def _summarize_bridge_file(path: str, label: str) -> None:
    obj = load_json(path)
    if not obj:
        print(f"  ({label}: file missing)")
        return
    results = obj.get("results") or []
    by_comp: dict[str, int] = {}
    high_anchor = []
    anchor_keys = {tuple(sorted([a, b])) for a, b, _ in ANCHOR_PAIRS}
    for r in results:
        b = r.get("bridge") or {}
        c = b.get("compatibility", "?")
        by_comp[c] = by_comp.get(c, 0) + 1
        key = tuple(sorted([r["a"]["arxiv_id"], r["b"]["arxiv_id"]]))
        if key in anchor_keys and c == "high":
            high_anchor.append((key, b.get("mediator")))
    print(f"  {label}: {len(results)} attempts, distribution={by_comp}")
    print(f"    anchor pairs rated 'high': {len(high_anchor)}")
    for key, med in high_anchor:
        print(f"      {key} -> mediator={med!r}")


def exp1_bridge_model_comparison() -> None:
    print("\n" + "=" * 70)
    print("EXP 1: Bridge-attempt model comparison")
    print("=" * 70)
    _summarize_bridge_file("bridge_attempts.json", "gemini-2.5-flash (original run 1)")
    _summarize_bridge_file("bridge_attempts_g35.json", "gemini-3.5-flash (exp 1)")


if __name__ == "__main__":
    exp2_full_corpus_ranking()
    exp1_bridge_model_comparison()

    # Also compare to the Claude subagent results from run 2 (different corpus
    # though, so apples-to-oranges on per-pair compatibility — just show the
    # aggregate).
    run2 = Path(__file__).resolve().parent.parent / "run2" / "bridge_attempts_claude.json"
    if run2.exists():
        print("\nClaude subagent reference (run 2, different corpus):")
        obj = json.loads(run2.read_text())
        results = obj.get("results") or []
        by_comp = {}
        for r in results:
            c = (r.get("bridge") or {}).get("compatibility", "?")
            by_comp[c] = by_comp.get(c, 0) + 1
        print(f"  Claude (run 2): {len(results)} attempts, distribution={by_comp}")
