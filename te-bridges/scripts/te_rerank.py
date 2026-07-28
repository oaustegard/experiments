"""Stage 6+7 — Slot embedding + asymmetric cosine rerank.

Embeds the asymmetric slot texts:
  - empirical papers: "phenomenon: ... regime: ... mechanism_unknown: ..."
  - theory papers:    "theorem_claim: ... regime: ... mechanism_provided: ..."

Then computes cross-axis cosine scores (only emp×th pairs, never emp×emp
or th×th) and outputs the top-N candidates for cheap-judge evaluation.

Output:
  data/te_slot_embs.json   — {arxiv_id: vector | null}
  data/te_reranked.json    — top-N pairs sorted by slot cosine sim

Usage:
  python te_rerank.py [--top-n 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import gemini_embed_batch, load_json, save_json  # noqa: E402

DEFAULT_TOP_N = 200


def slot_text(entry: dict | None) -> str:
    if not entry or not entry.get("slots"):
        return ""
    slots = entry["slots"]
    pool = entry.get("pool", "")
    parts = []
    if pool == "empirical":
        for k in ("phenomenon", "regime", "mechanism_unknown"):
            v = slots.get(k, "")
            if v and v.lower() not in ("none", "n/a", "not stated"):
                parts.append(f"{k}: {v}")
    else:
        for k in ("theorem_claim", "regime", "mechanism_provided"):
            v = slots.get(k, "")
            if v and v.lower() not in ("none", "n/a"):
                parts.append(f"{k}: {v}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = ap.parse_args()

    extractions = load_json("te_extractions.json", default={}) or {}
    candidates  = load_json("te_candidates.json")
    if not candidates:
        print("ERROR: run te_scan.py first", file=sys.stderr)
        sys.exit(1)

    # Build slot embeddings
    slot_embs = dict(load_json("te_slot_embs.json", default={}) or {})
    pending = [
        (aid, slot_text(entry))
        for aid, entry in extractions.items()
        if aid not in slot_embs and slot_text(entry)
    ]
    print(f"Embedding {len(pending)} slot texts ({len(slot_embs)} cached)…")
    if pending:
        vecs = gemini_embed_batch([t for _, t in pending])
        for (aid, _), v in zip(pending, vecs):
            slot_embs[aid] = v
        save_json("te_slot_embs.json", slot_embs)

    # Build indexed lookup for reranking
    pairs = candidates["pairs"]
    emp_ids = sorted({p["emp_arxiv"] for p in pairs if slot_embs.get(p["emp_arxiv"])})
    th_ids  = sorted({p["th_arxiv"]  for p in pairs if slot_embs.get(p["th_arxiv"])})
    print(f"Valid pairs: emp={len(emp_ids)}, th={len(th_ids)}")

    if not emp_ids or not th_ids:
        print("ERROR: no valid embeddings for rerank", file=sys.stderr)
        sys.exit(1)

    emp_idx = {a: i for i, a in enumerate(emp_ids)}
    th_idx  = {a: i for i, a in enumerate(th_ids)}

    E = np.asarray([slot_embs[a] for a in emp_ids], dtype=np.float32)
    T = np.asarray([slot_embs[a] for a in th_ids],  dtype=np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

    # Score only the candidate pairs (not the full cross product)
    scored: list[tuple[float, dict]] = []
    for p in pairs:
        ea = p["emp_arxiv"]
        ta = p["th_arxiv"]
        ei = emp_idx.get(ea)
        ti = th_idx.get(ta)
        if ei is None or ti is None:
            continue
        sim = float(E[ei] @ T[ti])
        scored.append((sim, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: args.top_n]

    out_pairs = []
    for sim, p in top:
        emp_entry = extractions.get(p["emp_arxiv"], {})
        th_entry  = extractions.get(p["th_arxiv"],  {})
        out_pairs.append({
            "slot_cosine": round(sim, 6),
            "specter_cosine": round(p.get("cosine_sim", 0.0), 6),
            "emp_arxiv": p["emp_arxiv"],
            "th_arxiv":  p["th_arxiv"],
            "emp_title": p.get("emp_title", ""),
            "th_title":  p.get("th_title",  ""),
            "emp_slots": (emp_entry.get("slots") or {}),
            "th_slots":  (th_entry.get("slots")  or {}),
        })

    save_json("te_reranked.json", {"n_pairs": len(out_pairs), "pairs": out_pairs})
    print(f"Done: {len(out_pairs)} pairs written to data/te_reranked.json")


if __name__ == "__main__":
    main()
