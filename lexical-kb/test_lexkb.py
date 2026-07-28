#!/usr/bin/env python3
"""Three-way retrieval comparison on the tiny corpus: raw BM25 vs RM3 vs
agent-expansion. Demonstrates that the embedding-free KB's quality rides on the
query-expansion step the SKILL.md protocol mandates.

Each query has a known gold document (by source_path). We report the best rank
at which *any* chunk of the gold doc appears, under each mode. Lower is better;
"—" means not retrieved in the top-k scanned.

Run: python3 test_lexkb.py [--target-chars N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "skill_template"))
import build_lexkb as B  # noqa: E402
from search import Index, build_query, rm3_expand, tokenize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke  # noqa: E402

CORPUS = spoke("remax_kb") / "examples/tiny_corpus"

# (label, raw query, gold source_path, agent core terms, agent expansion terms)
CASES = [
    (
        "plural/synonym",
        "political factions",
        "federalist_10_factions.txt",
        ["factions"],
        ["faction", "parties", "party", "special interests", "majority", "minority"],
    ),
    (
        "separation of powers",
        "separation of powers between branches of government",
        "federalist_51_checks.txt",
        ["separation", "powers", "branches"],
        ["checks", "balances", "departments", "legislative", "executive",
         "judiciary", "encroachments", "partition"],
    ),
    (
        "property/inequality",
        "economic inequality and property rights",
        "federalist_10_property.txt",
        ["property", "inequality"],
        ["faculties", "unequal", "wealth", "rights", "distribution", "classes"],
    ),
    (
        "paraphrase (hard)",
        "honoring soldiers who died in battle",
        "gettysburg.txt",
        ["soldiers", "died", "battle"],
        ["brave men", "gave their lives", "fallen", "dead", "consecrate",
         "devotion", "sacrifice", "war", "battlefield"],
    ),
]


def best_rank(hits_doc_paths: list[str], gold: str) -> str:
    for i, p in enumerate(hits_doc_paths, 1):
        if p == gold:
            return str(i)
    return "—"


def ranked_docpaths(index: Index, query: dict, k: int) -> list[str]:
    scores = index.score(query)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [index.chunks[i]["meta"]["source_path"] for i, _ in ranked]


def gold_margin(index: Index, query: dict, gold: str) -> float:
    """Normalized confidence: (best gold-chunk score − best distractor score) /
    best gold score. >0 means gold leads; higher = more separable. Has
    resolution even when rank saturates at 1 on a tiny, disjoint corpus."""
    scores = index.score(query)
    best_gold = max((s for d, s in scores.items()
                     if index.chunks[d]["meta"]["source_path"] == gold), default=0.0)
    best_other = max((s for d, s in scores.items()
                      if index.chunks[d]["meta"]["source_path"] != gold), default=0.0)
    if best_gold <= 0:
        return -1.0
    return (best_gold - best_other) / best_gold


def run(target_chars: int) -> None:
    out = HERE / "out" / f"kb_{target_chars}"
    chunks = B.collect_chunks(CORPUS, {"txt"}, target_chars, 40)
    index_dict = B.build_index(chunks, 1.5, 0.75)
    B.write_bundle(out, chunks, index_dict, "tiny corpus")
    index = Index(out)
    k = max(5, index.N)  # scan all chunks for rank measurement

    print(f"\n=== target_chars={target_chars}  ({index.N} chunks, "
          f"avgdl={index.avgdl:.0f} tok) ===")
    print(f"{'case':<22}{'rank raw/rm3/exp':>18}{'margin raw→exp':>18}")
    print("-" * 60)
    for label, raw_q, gold, core, expand in CASES:
        raw_query = build_query([raw_q], [], 1.0, 0.4)
        exp_query = build_query(core, expand, 1.0, 0.4, backstop=[raw_q])
        rm3_query = rm3_expand(index, raw_query, n_docs=4, n_terms=12)
        ranks = "/".join((
            best_rank(ranked_docpaths(index, raw_query, k), gold),
            best_rank(ranked_docpaths(index, rm3_query, k), gold),
            best_rank(ranked_docpaths(index, exp_query, k), gold),
        ))
        m_raw = gold_margin(index, raw_query, gold)
        m_exp = gold_margin(index, exp_query, gold)
        print(f"{label:<22}{ranks:>18}{f'{m_raw:+.2f} → {m_exp:+.2f}':>18}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-chars", type=int, default=None,
                    help="single size; default sweeps a few")
    args = ap.parse_args()
    sizes = [args.target_chars] if args.target_chars is not None else [500, 1200, 0]
    print("Rank of gold document (lower = better; '—' = not retrieved)")
    print("Modes: raw=question as-is | rm3=pseudo-relevance feedback | "
          "expand=agent core+expansion")
    for s in sizes:
        run(s)
    print("\n(target_chars=0 means whole-document chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
