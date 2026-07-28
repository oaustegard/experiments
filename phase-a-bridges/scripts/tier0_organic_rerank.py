"""Tier 0 — Organic-pairs rerank, no anchor force-include.

Question: at the existing 997-paper corpus, does the cascade discover
anything that isn't on our anchor force-include list?

Pipeline:
  1. Take top-100 candidate pairs from data/candidate_pairs.json
  2. DROP all anchor pairs (kind=='anchor')
  3. From the remainder, take top-30 by SPECTER2 normalized Hamming
  4. Slot-extract for any papers not already in exp3_extracts.json (batch
     concurrent gemini-3.5-flash calls — there's no batch generateContent,
     but we can run many in parallel)
  5. Slot-embed via :batchEmbedContents (already 100x batched)
  6. Recompute pair cosines in slot space, sort, output top-20
  7. The Claude subagent bridge step is run by the orchestrator (main
     session), not this script -- it writes prompts to data/tier0_prompts.json

Outputs:
  data/tier0_candidates.json  — top-30 organic pairs (slot-cosine ranked)
  data/tier0_extracts.json    — slot extractions for the involved papers
  data/tier0_slot_embs.json   — slot embeddings
  data/tier0_prompts.json     — Claude prompts ready for Agent-tool dispatch
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gemini_embed_batch, gemini_generate, load_json, save_json  # noqa: E402
from exp_structured_extract import EXTRACT_PROMPT, slot_text  # noqa: E402
from stage5_claude_subagent import SUBAGENT_PROMPT  # noqa: E402


N_FROM_BAND = 30
N_TO_CLAUDE = 20
EXTRACT_PARALLELISM = 8


def _parse_json_lax(s: str) -> dict | None:
    s = s.strip()
    s = s.lstrip("`").lstrip("json").lstrip()
    s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def extract_one(arxiv_id: str, title: str, body: str) -> tuple[str, dict | None]:
    if not body or len(body) < 200:
        return arxiv_id, None
    prompt = EXTRACT_PROMPT.format(title=title, body=body[:6000])
    raw = gemini_generate(prompt, json_mode=True, max_tokens=600, thinking_budget=0)
    return arxiv_id, _parse_json_lax(raw)


def main() -> None:
    cps = load_json("candidate_pairs.json")
    if not cps:
        print("ERROR: candidate_pairs.json missing", file=sys.stderr)
        sys.exit(1)

    organic = [p for p in cps["pairs"] if p.get("kind") != "anchor"]
    organic.sort(key=lambda p: p["distance_norm"])
    top = organic[:N_FROM_BAND]
    print(f"Tier 0: {len(top)} organic candidates from SPECTER2 band (drop forced anchors)")

    needed_ids = sorted({p["a"]["arxiv_id"] for p in top} | {p["b"]["arxiv_id"] for p in top})
    print(f"  unique papers: {len(needed_ids)}")

    # Pull metadata + body texts
    meta = load_json("metadata.json")
    by_arxiv = {m["arxiv_id"]: m for m in meta}
    body_texts = load_json("body_texts.json", default={}) or {}
    full_body_texts = load_json("full_body_texts.json", default={}) or {}
    for k, v in full_body_texts.items():
        body_texts.setdefault(k, v)

    # Reuse exp3 extracts where available
    extracts = load_json("exp3_extracts.json", default={}) or {}
    # Reuse tier0 cache too (idempotent)
    extracts.update(load_json("tier0_extracts.json", default={}) or {})

    # --- Concurrent extraction --------------------------------------------
    pending = [
        aid for aid in needed_ids
        if (aid not in extracts or not extracts[aid]) and body_texts.get(aid)
    ]
    print(f"  cached extracts: {len(needed_ids) - len(pending)}, pending: {len(pending)}")
    if pending:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=EXTRACT_PARALLELISM) as pool:
            futs = {
                pool.submit(
                    extract_one,
                    aid,
                    (by_arxiv.get(aid) or {}).get("title", ""),
                    body_texts.get(aid, ""),
                ): aid
                for aid in pending
            }
            for fut in as_completed(futs):
                aid, ext = fut.result()
                extracts[aid] = ext
                print(f"  extract {aid}: {'ok' if ext else 'fail'}")
        print(f"  extraction batch done in {time.time() - t0:.1f}s")
    save_json("tier0_extracts.json", extracts)

    # --- Batch slot embedding ---------------------------------------------
    slot_embs = load_json("tier0_slot_embs.json", default={}) or {}
    embed_ids: list[str] = []
    embed_texts: list[str] = []
    for aid in needed_ids:
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
        embed_ids.append(aid)
        embed_texts.append(text)
    if embed_texts:
        print(f"  batch-embedding {len(embed_texts)} slot texts…")
        t0 = time.time()
        vecs = gemini_embed_batch(embed_texts)
        for aid, v in zip(embed_ids, vecs):
            slot_embs[aid] = v
        print(f"  done in {time.time() - t0:.1f}s ({sum(1 for v in vecs if v)}/{len(vecs)} valid)")
    save_json("tier0_slot_embs.json", slot_embs)

    # --- Rerank pairs by slot cosine --------------------------------------
    rescored: list[dict] = []
    for p in top:
        a, b = p["a"]["arxiv_id"], p["b"]["arxiv_id"]
        va, vb = slot_embs.get(a), slot_embs.get(b)
        if not va or not vb:
            continue
        ua = np.asarray(va, dtype=np.float32)
        ub = np.asarray(vb, dtype=np.float32)
        cos_sim = float((ua @ ub) / (np.linalg.norm(ua) * np.linalg.norm(ub)))
        rescored.append({**p, "slot_cosine_dist": round(1 - cos_sim, 6)})
    rescored.sort(key=lambda x: x["slot_cosine_dist"])
    out = rescored[:N_TO_CLAUDE]
    save_json("tier0_candidates.json", {"n_input": len(top), "n_with_embeddings": len(rescored), "pairs": out})

    print(f"\nTop {len(out)} organic pairs by slot-cosine (no anchors):")
    for i, p in enumerate(out, 1):
        a = p["a"]; b = p["b"]
        print(f"  {i:2d}. cos={p['slot_cosine_dist']:.4f}  ham={p['distance_norm']:.4f}  "
              f"{a['arxiv_id']} ↔ {b['arxiv_id']}  [{p['kind']}]")
        print(f"      A: {(a.get('title') or '')[:75]}")
        print(f"      B: {(b.get('title') or '')[:75]}")

    # --- Build Claude prompts for orchestrator -----------------------------
    prompts = []
    for p in out:
        a, b = p["a"], p["b"]
        a_body = body_texts.get(a["arxiv_id"], "")
        b_body = body_texts.get(b["arxiv_id"], "")
        prompt = SUBAGENT_PROMPT.format(
            a_arxiv=a["arxiv_id"],
            a_sub=a.get("subfield", "?"),
            a_title=a["title"],
            a_body=a_body[:6000],
            b_arxiv=b["arxiv_id"],
            b_sub=b.get("subfield", "?"),
            b_title=b["title"],
            b_body=b_body[:6000],
            hamming=p["distance_norm"],
            body=p["slot_cosine_dist"],
        )
        prompts.append({
            "pair_key": f"{a['arxiv_id']}__{b['arxiv_id']}",
            "a_arxiv": a["arxiv_id"],
            "b_arxiv": b["arxiv_id"],
            "prompt": prompt,
        })
    save_json("tier0_prompts.json", prompts)
    print(f"\nWrote {len(prompts)} Claude prompts to data/tier0_prompts.json")
    print("Orchestrator (main session) will dispatch via Agent tool.")


if __name__ == "__main__":
    main()
