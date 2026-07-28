"""Twin diagnostic for path C cross-domain run.

For each anchor in anchors.json:
  1. Read the SPECTER2-cosine-ranked candidate list (te_candidates.json)
  2. Read expected_twin_arxiv from the anchor config
  3. Report:
     - Whether each expected twin appears anywhere in the candidate union
     - Its rank if so (lower = nearer the anchor by SPECTER2 cosine)
     - The 'co-conversation density' of the top-K: fraction of titles
       that look like the twin conversation (rough heuristic via keyword
       lists derived from expected_twin_note)
     - The raw top-K title list, for eyeball judgement

Diagnostic verdict:
  - 'TWIN-COUSIN' if at least one expected_twin_arxiv appears in top-K
    OR the keyword density in top-K is >= 20%
  - 'TWIN-MISS' otherwise

A majority TWIN-COUSIN across the 9 anchors means SPECTER2 geometry
contains bridge signal at small scale → Path B becomes defensible.
A majority TWIN-MISS means the SLT objection is doing real work.

Run from $TE_DATA_DIR with anchors.json + te_candidates.json present.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


KW_HINTS = {
    # Crude keyword bag per twin conversation. Matched against candidate
    # titles only (cheap). Tuned to be specific enough to avoid false
    # positives but broad enough to catch close cousins.
    "rmt_scaling": [
        "random matrix", "marchenko", "wigner", "spectral density",
        "heavy-tail", "spectrum", "eigenvalue", "high-dimensional regression",
        "neural scaling", "scaling law",
    ],
    "isoperimetric_adv": [
        "isoperimetric", "concentration of measure", "talagrand",
        "lipschitz", "adversarial robust", "geometry of high-dim",
        "spherical", "geodesic", "manifold robustness",
    ],
    "max_margin": [
        "max-margin", "margin bound", "logistic", "boosting margin",
        "support vector", "rkhs", "rademacher", "kernel margin",
        "exponentiated gradient",
    ],
    "stat_mech_dd": [
        "statistical mechanics", "replica", "spin glass", "high-dim ridge",
        "double descent", "interpolation peak", "min-norm", "overparameterized",
        "free energy", "replica symmetry",
    ],
    "slt_phase": [
        "singular learning", "watanabe", "phase transition",
        "criticality", "free energy", "rlct", "real log canonical",
        "loss landscape", "grokking",
    ],
    "compressed_sensing": [
        "compressed sensing", "compressive sensing", "sparse recovery",
        "restricted isometry", "rip", "l1 minimization", "basis pursuit",
        "lasso", "sparse signal", "incoherence",
    ],
    "lottery_ticket_pruning": [
        "lottery ticket", "pruning", "sparse network", "winning ticket",
        "magnitude pruning", "subnetwork", "iterative pruning",
        "neural network sparsification",
    ],
    "solomonoff_bayes": [
        "solomonoff", "kolmogorov complexity", "universal induction",
        "aixi", "algorithmic probability", "bayes optimal",
        "minimum description length", "mdl", "pac-bayes",
    ],
    "icl_bayes": [
        "in-context learning", "few-shot", "implicit bayesian",
        "transformer learn", "function class", "meta-learning",
        "context length", "prompt engineering",
    ],
}


# Per-anchor twin keyword bag(s). An anchor matches twin signal if any
# title hits any of its bags' keywords.
ANCHOR_KW = {
    "2001.08361": ["rmt_scaling"],          # Kaplan → RMT
    "1706.06083": ["isoperimetric_adv"],    # Madry → isoperimetric
    "1710.10345": ["max_margin"],           # Soudry → max-margin
    "1812.11118": ["stat_mech_dd"],         # Belkin → stat mech
    "2201.02177": ["slt_phase"],            # Power → SLT
    "1803.03635": ["compressed_sensing"],   # Frankle → compressed sensing
    "math/0502327": ["lottery_ticket_pruning"],  # CRT → lottery
    "2208.01066": ["solomonoff_bayes"],     # Garg → Solomonoff
    "cs/0701125": ["icl_bayes"],            # Hutter → ICL
}


def load(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def normalize(s: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", (s or "").lower())


def kw_hit(title: str, bags: list[str]) -> str | None:
    t = normalize(title)
    for b in bags:
        for k in KW_HINTS.get(b, []):
            if k in t:
                return f"{b}:'{k}'"
    return None


def main() -> None:
    data_dir = Path(os.environ.get("TE_DATA_DIR", "."))
    anchors_path = data_dir / "anchors.json"
    cands_path   = data_dir / "te_candidates.json"
    raw_path     = data_dir / "anchor_candidates_filtered.json"

    if not anchors_path.exists() or not cands_path.exists():
        print(f"ERROR: need {anchors_path} and {cands_path}", file=sys.stderr)
        sys.exit(2)

    cfg = load(anchors_path)
    cands = load(cands_path)
    raw = load(raw_path) if raw_path.exists() else {}

    # Index pairs by source_anchor for ranking
    by_anchor: dict[str, list[dict]] = {}
    for p in cands.get("pairs", []):
        by_anchor.setdefault(p["source_anchor"], []).append(p)
    for aid in by_anchor:
        by_anchor[aid].sort(key=lambda x: -x["cosine_sim"])

    # Also index the post-filter candidate union per anchor (which may include
    # twins that didn't make it into the kept top-K)
    union_by_anchor: dict[str, list[dict]] = {}
    for aid, blk in raw.items():
        union_by_anchor[aid] = blk.get("candidates", [])

    print("=" * 76)
    print("PATH C CROSS-DOMAIN — TWIN DIAGNOSTIC")
    print("=" * 76)

    twin_cousin = 0
    twin_miss   = 0

    for anchor in cfg["anchors"]:
        aid     = anchor["arxiv_id"]
        label   = anchor["label"]
        twins   = anchor.get("expected_twin_arxiv", [])
        pairs   = by_anchor.get(aid, [])
        union   = union_by_anchor.get(aid, [])
        bags    = ANCHOR_KW.get(aid, [])

        print(f"\n[{aid}] {label}")
        print(f"  twin bags: {bags}")
        print(f"  union size: {len(union)} candidates, top-K kept: {len(pairs)}")

        # Twin hits in union (regardless of top-K)
        union_arx = {c["arxiv_id"] for c in union}
        twin_in_union = [t for t in twins if t in union_arx]
        # Twin hits in top-K
        topk_arx = []
        for p in pairs:
            topk_arx.append(p["emp_arxiv"] if p["emp_arxiv"] != aid else p["th_arxiv"])
        twin_in_topk = [t for t in twins if t in topk_arx]

        # Rank of twin in top-K if present
        for t in twin_in_topk:
            for i, x in enumerate(topk_arx, start=1):
                if x == t:
                    print(f"  TWIN IN TOP-K  rank={i:3d}  arxiv={t}")
                    break
        for t in twin_in_union:
            if t not in twin_in_topk:
                print(f"  twin in union (NOT top-K): {t}")
        for t in twins:
            if t not in union_arx:
                print(f"  twin missed entirely:       {t}")

        # Keyword density in top-K
        hits = []
        for p in pairs[:40]:  # top 40 only
            t_arx = p["emp_arxiv"] if p["emp_arxiv"] != aid else p["th_arxiv"]
            t_title = p["emp_title"] if p["emp_arxiv"] != aid else p["th_title"]
            hit = kw_hit(t_title, bags) if bags else None
            if hit:
                hits.append((t_arx, hit, t_title[:80]))

        kw_density = len(hits) / max(1, len(pairs[:40]))
        print(f"  top-40 keyword density (twin-bag): {len(hits)}/{min(40, len(pairs))} = {kw_density:.0%}")
        for arx, why, title in hits[:6]:
            print(f"    + {arx:18s}  {why:30s}  {title}")

        # Verdict
        if twin_in_topk or kw_density >= 0.20:
            print(f"  VERDICT: TWIN-COUSIN")
            twin_cousin += 1
        elif twin_in_union:
            print(f"  VERDICT: TWIN-MISS (twin in union but not top-K; SPECTER2 ranked it below folklore)")
            twin_miss += 1
        else:
            print(f"  VERDICT: TWIN-MISS (no twin in union, low keyword density)")
            twin_miss += 1

    print("\n" + "=" * 76)
    print(f"SUMMARY: TWIN-COUSIN={twin_cousin}  TWIN-MISS={twin_miss}")
    total = twin_cousin + twin_miss
    if twin_cousin >= total - twin_cousin:
        print("PATH C VERDICT: SPECTER2 geometry holds bridge signal at small scale.")
        print("Path B (1.9M, $435) becomes defensible.")
    else:
        print("PATH C VERDICT: SPECTER2 only surfaces same-conversation folklore.")
        print("SLT objection is doing real work; the cascade mechanism needs to")
        print("change before the corpus does. Path B does NOT become defensible.")


if __name__ == "__main__":
    main()
