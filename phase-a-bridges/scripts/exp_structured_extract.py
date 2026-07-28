"""Experiment 3 — Structured-extract embeddings vs raw-body embeddings.

Hypothesis: the HMR↔Pach-Raz blind spot in dense embedding space is a
surface-vocabulary mismatch. Extract a 4-slot record (problem / methods
/ regime / hypothesized_generalization) for each paper via gemini-3.5-
flash, concatenate the slot strings, embed via gemini-embedding-001,
recompute pairwise cosines. If the bridge pair's rank drops measurably
vs the raw-body baseline, phase B's coarse stage should be slot-based.

Sample: 10 phase-0 anchors + 40 random non-anchor papers from the run-1
corpus. Already have body texts cached (`body_texts.json` + spillover
from `full_body_texts.json`). Cost: ~$0.02 in Gemini calls. Wall: ~3 min.

Outputs:
  data/exp3_extracts.json     — {arxiv_id: {problem, methods, regime, gen}}
  data/exp3_slot_embeddings.json — {arxiv_id: vector}
  data/exp3_results.json      — rank table + summary
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gemini_embed_batch, gemini_generate, load_json, save_json  # noqa: E402
from exp_analyze import ANCHOR_PAIRS  # noqa: E402


ANCHOR_IDS = sorted({a for a, _, _ in ANCHOR_PAIRS} | {b for _, b, _ in ANCHOR_PAIRS})
N_RANDOM = 40
SEED = 1234

EXTRACT_PROMPT = """You are extracting structured fields from a research paper for cross-field similarity search.

Title: {title}
Body snippet:
\"\"\"
{body}
\"\"\"

Return a JSON object with exactly these four keys. Aim for sentences a domain-expert reader would write, with concrete nouns over jargon when both are accurate. The extracted text will be embedded; the goal is for two papers with the same DEEP problem structure to land near each other in embedding space even when their surface vocabulary differs.

- problem:               one sentence stating the problem the paper solves (focus on the abstract structure of the problem, not the field-specific framing)
- methods:               one sentence naming the main techniques and the kind of mathematical objects they operate on
- regime:                one sentence stating the scale/setting where the result applies
- hypothesized_generalization: one sentence speculating what generalization the result hints at (or "none")

Output ONLY the JSON object."""


def extract_paper(arxiv_id: str, title: str, body: str) -> dict | None:
    if not body or len(body) < 200:
        return None
    prompt = EXTRACT_PROMPT.format(title=title, body=body[:6000])
    raw = gemini_generate(prompt, json_mode=True, max_tokens=600, thinking_budget=0)
    s = raw.strip().lstrip("`").lstrip("json").lstrip()
    s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def slot_text(ext: dict) -> str:
    parts = []
    for k in ("problem", "methods", "regime", "hypothesized_generalization"):
        v = ext.get(k, "")
        if v and v.lower() not in ("none", "n/a"):
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


def main() -> None:
    meta = load_json("metadata.json")
    body_texts = load_json("body_texts.json", default={}) or {}
    full_body_texts = load_json("full_body_texts.json", default={}) or {}
    for k, v in full_body_texts.items():
        body_texts.setdefault(k, v)

    by_arxiv = {m["arxiv_id"]: m for m in meta}

    # Sample: anchors + random others (with body available)
    rng = random.Random(SEED)
    pool = [a for a in by_arxiv if a not in ANCHOR_IDS and body_texts.get(a)]
    random_sample = rng.sample(pool, min(N_RANDOM, len(pool)))
    sample = list(ANCHOR_IDS) + random_sample
    print(f"Exp 3 corpus: {len(ANCHOR_IDS)} anchors + {len(random_sample)} random = {len(sample)} papers")

    # --- Extract slots ----------------------------------------------------
    extracts = load_json("exp3_extracts.json", default={}) or {}
    for i, aid in enumerate(sample, 1):
        if aid in extracts and extracts[aid]:
            continue
        body = body_texts.get(aid, "")
        title = (by_arxiv.get(aid) or {}).get("title", "")
        ext = extract_paper(aid, title, body)
        extracts[aid] = ext
        ok = ext is not None
        print(f"  extract [{i}/{len(sample)}] {aid}: {'ok' if ok else 'fail'} body_len={len(body)}")
        if i % 10 == 0:
            save_json("exp3_extracts.json", extracts)
        time.sleep(0.3)
    save_json("exp3_extracts.json", extracts)

    # --- Embed slots (batched) -------------------------------------------
    slot_embs = load_json("exp3_slot_embeddings.json", default={}) or {}
    pending: list[tuple[str, str]] = []
    for aid in sample:
        if aid in slot_embs and slot_embs[aid]:
            continue
        ext = extracts.get(aid)
        if not ext:
            slot_embs[aid] = None
            continue
        text = slot_text(ext)
        if not text:
            slot_embs[aid] = None
            continue
        pending.append((aid, text))
    if pending:
        print(f"  batch-embedding {len(pending)} slot texts…")
        t0 = time.time()
        vecs = gemini_embed_batch([t for _, t in pending])
        for (aid, _), v in zip(pending, vecs):
            slot_embs[aid] = v
        print(f"  batch done in {time.time() - t0:.2f}s, "
              f"{sum(1 for v in vecs if v)}/{len(vecs)} valid")
    save_json("exp3_slot_embeddings.json", slot_embs)

    # --- Rank anchor pairs in slot-embedding space ------------------------
    valid = [a for a in sample if slot_embs.get(a)]
    X = np.asarray([slot_embs[a] for a in valid], dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    D = 1 - S
    np.fill_diagonal(D, np.inf)
    n = len(valid)
    idx = {k: i for i, k in enumerate(valid)}
    iu, ju = np.triu_indices(n, k=1)
    all_d = D[iu, ju]
    print(f"\nValid papers: {n}, pairs: {len(all_d)}")

    # Also load raw-body embeddings for comparison.
    raw_embs = load_json("full_body_embeddings.json", default={}) or load_json("body_embeddings.json", default={}) or {}
    raw_valid = [a for a in sample if raw_embs.get(a)]
    if raw_valid:
        XR = np.asarray([raw_embs[a] for a in raw_valid], dtype=np.float32)
        XR = XR / np.linalg.norm(XR, axis=1, keepdims=True)
        SR = XR @ XR.T
        DR = 1 - SR
        np.fill_diagonal(DR, np.inf)
        idx_r = {k: i for i, k in enumerate(raw_valid)}
        iu_r, ju_r = np.triu_indices(len(raw_valid), k=1)
        all_d_r = DR[iu_r, ju_r]

    rows = []
    print(f"\n{'pair':30s}  {'slot rank':>20s}  {'raw rank':>20s}")
    for a, b, name in ANCHOR_PAIRS:
        if a not in idx or b not in idx:
            rows.append({"pair": name, "slot_rank": None, "raw_rank": None})
            print(f"  {name:30s}  (missing in slot set)")
            continue
        i, j = sorted([idx[a], idx[b]])
        d = float(D[i, j])
        rank_s = int((all_d < d).sum())
        pct_s = 100 * rank_s / len(all_d)
        rank_r = pct_r = None
        if raw_valid and a in idx_r and b in idx_r:
            ir, jr = sorted([idx_r[a], idx_r[b]])
            dr = float(DR[ir, jr])
            rank_r = int((all_d_r < dr).sum())
            pct_r = 100 * rank_r / len(all_d_r)
        rows.append({
            "pair": name,
            "slot_cos_dist": round(d, 4),
            "slot_rank": rank_s,
            "slot_total": len(all_d),
            "slot_pct": round(pct_s, 2),
            "raw_rank": rank_r,
            "raw_pct": round(pct_r, 2) if pct_r is not None else None,
        })
        slot_str = f"{rank_s}/{len(all_d)} ({pct_s:.1f}%)"
        raw_str = f"{rank_r}/{len(all_d_r)} ({pct_r:.1f}%)" if pct_r is not None else "—"
        print(f"  {name:30s}  {slot_str:>20s}  {raw_str:>20s}")

    save_json("exp3_results.json", {
        "sample_size": n,
        "n_anchors_in_sample": sum(1 for a in ANCHOR_IDS if a in idx),
        "anchor_pair_ranks": rows,
    })

    # Top 10 closest in slot space
    print(f"\nTop 10 closest pairs in slot space:")
    order = all_d.argsort()
    anchor_keys = {tuple(sorted([a, b])) for a, b, _ in ANCHOR_PAIRS}
    for k in range(10):
        p = order[k]
        i, j = iu[p], ju[p]
        key = tuple(sorted([valid[i], valid[j]]))
        marker = " ANCHOR" if key in anchor_keys else ""
        print(f"  {k+1:2d}. cos={all_d[p]:.4f}  {valid[i]} ↔ {valid[j]}{marker}")


if __name__ == "__main__":
    main()
