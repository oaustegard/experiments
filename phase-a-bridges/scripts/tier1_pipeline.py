"""Tier 1 — Filter, slot-extract, rerank, cheap-judge.

Runs AFTER tier1_assemble.py (which builds the corpus + SPECTER2 codes).
Produces Claude bridge prompts for the top-20 candidates.

Stages:
  3. Band scan + author/sequential dedup -> top-K cross-field pairs
  4. Body fetch (concurrent, polite) for unique band papers
  5. Slot extraction (concurrent gemini-3.5-flash)
  6. Slot embedding (batched :batchEmbedContents)
  7. Slot-cosine rerank -> top-N
  8. Cheap-LLM-judge via gemini-3.5-flash (yes/no per pair) -> top-20
  9. Write Claude prompts to data/tier1_bridge_prompts.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA,
    gemini_embed_batch,
    gemini_generate,
    load_json,
    save_json,
)
from exp_structured_extract import EXTRACT_PROMPT, slot_text  # noqa: E402
from remax import hamming_pairs_chunked, normalized_hamming  # noqa: E402
from stage3_scan import (  # noqa: E402
    CS_THEORY_CATS,
    MATH_CATS,
    fetch_arxiv_categories,
    primary_field,
    subfield,
)
from stage4_rerank import body_window, fetch_body  # noqa: E402
from stage5_claude_subagent import SUBAGENT_PROMPT  # noqa: E402

BAND_LO, BAND_HI = 0.14, 0.24
TOP_K_PAIRS_FROM_BAND = 5000
TOP_N_AFTER_SLOT = 200
TOP_M_AFTER_JUDGE = 20

EXTRACT_PARALLELISM = 6   # 2.5-flash has higher RPS budget than 3.5
BODY_FETCH_PARALLELISM = 8
JUDGE_PARALLELISM = 2      # judge uses 3.5-flash; keep tight

JUDGE_PROMPT = """You are filtering pairs of research papers for cross-field bridge potential.

Paper A — {a_arxiv}
Title: {a_title}
Problem: {a_problem}
Methods: {a_methods}

Paper B — {b_arxiv}
Title: {b_title}
Problem: {b_problem}
Methods: {b_methods}

These papers come from different subfields. Is there a plausible deep structural connection between them that a competent researcher could attempt to bridge — not requiring shared surface vocabulary, but sharing a mathematical mediator concept (e.g. lattice constructions, Galois cohomology, Fourier analysis, asymptotic spectrum, tree decomposition, belief propagation, etc.)?

Answer with ONE JSON object:
{{
  "verdict": "promising" | "weak" | "unrelated",
  "mediator_guess": "short phrase or 'none'"
}}

Be strict — "promising" means you can name a specific mathematical concept that genuinely bridges both papers; "weak" means there's some surface similarity but no clear bridge concept; "unrelated" means no plausible connection.

Output ONLY the JSON object."""


# -------------------- Stage 3: band scan + dedup ----------------------------

def author_sequential_keys(meta_by_idx: dict[int, dict]) -> dict[str, list[int]]:
    """Bucket papers by likely-author-cluster key: arXiv YYMM prefix.
    Sequential arXiv IDs from the same month are very often same-author."""
    buckets: dict[str, list[int]] = {}
    for i, m in meta_by_idx.items():
        aid = m.get("arxiv_id", "")
        prefix = aid.split(".")[0] if "." in aid else aid[:4]
        buckets.setdefault(prefix, []).append(i)
    return buckets


def is_sequential_pair(a: str, b: str) -> bool:
    """Same YYMM prefix + numeric IDs within 50 of each other."""
    if "." not in a or "." not in b:
        return False
    pa, na = a.split(".", 1)
    pb, nb = b.split(".", 1)
    if pa != pb:
        return False
    try:
        return abs(int(na) - int(nb)) <= 50
    except ValueError:
        return False


def band_scan_with_dedup() -> list[dict]:
    meta = load_json("metadata.json")
    cm = load_json("codes_meta.json")
    codes = np.frombuffer((DATA / "codes.bin").read_bytes(), dtype=np.uint8).reshape(
        cm["n"], cm["bytes_per_vec"]
    )
    bits = cm["bits_per_vec"]
    print(f"Stage 3: band scan over {cm['n']} papers (band [{BAND_LO}, {BAND_HI}])")

    # Enrich with arXiv categories.
    cats_map = load_json("arxiv_cats.json")
    if not cats_map:
        print("  fetching arXiv categories…")
        cats_map = fetch_arxiv_categories([m["arxiv_id"] for m in meta if m.get("arxiv_id")])
        save_json("arxiv_cats.json", cats_map)
    for m in meta:
        cats = cats_map.get(m.get("arxiv_id"), [])
        m["arxiv_cats"] = cats
        m["field"] = primary_field(cats)
        m["subfield"] = subfield(cats)

    math_idx = [i for i, m in enumerate(meta) if m["field"] == "math"]
    cs_idx = [i for i, m in enumerate(meta) if m["field"] == "cs"]
    print(f"  partition: math={len(math_idx)} cs={len(cs_idx)} other={len(meta) - len(math_idx) - len(cs_idx)}")

    # math x cs band pairs.
    print("  computing math×cs Hamming…")
    A = codes[math_idx]
    B = codes[cs_idx]
    D = hamming_pairs_chunked(A, B)
    Dn = normalized_hamming(D, bits)
    mask = (Dn >= BAND_LO) & (Dn <= BAND_HI)
    ii, jj = np.where(mask)
    print(f"  math×cs in band: {len(ii)} pairs")

    # Intra-math cross-subfield (math.NT × math.CO is the Erdős axis).
    by_sub: dict[str, list[int]] = {}
    for i in math_idx:
        by_sub.setdefault(meta[i]["subfield"], []).append(i)

    intra_math_pairs: list[tuple[float, int, int, str]] = []
    sub_keys = sorted(by_sub)
    for ia, sa in enumerate(sub_keys):
        for sb in sub_keys[ia + 1:]:
            la = by_sub[sa]
            lb = by_sub[sb]
            if not la or not lb:
                continue
            Da = codes[la]
            Db = codes[lb]
            Dab = hamming_pairs_chunked(Da, Db)
            Dnab = normalized_hamming(Dab, bits)
            mm = (Dnab >= BAND_LO) & (Dnab <= BAND_HI)
            for r, c in zip(*np.where(mm)):
                intra_math_pairs.append((float(Dnab[r, c]), la[r], lb[c], f"{sa}×{sb}"))
    print(f"  intra-math cross-subfield in band: {len(intra_math_pairs)} pairs")

    # Aggregate.
    candidates: list[tuple[float, int, int, str]] = []
    for r, c in zip(ii, jj):
        candidates.append((float(Dn[r, c]), math_idx[r], cs_idx[c], "math×cs"))
    candidates.extend(intra_math_pairs)

    # Dedup: drop sequential-arXiv pairs (proxy for same author group).
    pre_dedup = len(candidates)
    candidates = [
        (d, i, j, k) for (d, i, j, k) in candidates
        if not is_sequential_pair(meta[i]["arxiv_id"], meta[j]["arxiv_id"])
    ]
    print(f"  after sequential-arXiv dedup: {len(candidates)} pairs (dropped {pre_dedup - len(candidates)})")

    # Sort + take top K.
    candidates.sort(key=lambda x: x[0])
    top = candidates[:TOP_K_PAIRS_FROM_BAND]
    out = []
    for dist_norm, i, j, kind in top:
        out.append({
            "distance_norm": round(dist_norm, 6),
            "kind": kind,
            "a": {
                "arxiv_id": meta[i]["arxiv_id"],
                "title": meta[i]["title"],
                "subfield": meta[i]["subfield"],
            },
            "b": {
                "arxiv_id": meta[j]["arxiv_id"],
                "title": meta[j]["title"],
                "subfield": meta[j]["subfield"],
            },
        })
    save_json("candidate_pairs.json", {"n_pairs": len(out), "band": [BAND_LO, BAND_HI], "pairs": out})
    print(f"  saved top {len(out)} candidate pairs")
    return out


# -------------------- Stage 4: body fetch (concurrent) ---------------------

def fetch_bodies_concurrent(arxiv_ids: list[str]) -> dict[str, str]:
    cache = load_json("body_texts.json", default={}) or {}
    pending = [a for a in arxiv_ids if a not in cache]
    print(f"  body fetch: {len(cache)} cached, {len(pending)} pending")
    if not pending:
        return cache

    def _fetch(aid: str) -> tuple[str, str]:
        body = fetch_body(aid)
        return aid, body_window(body)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=BODY_FETCH_PARALLELISM) as pool:
        futs = {pool.submit(_fetch, a): a for a in pending}
        done = 0
        for fut in as_completed(futs):
            aid, text = fut.result()
            cache[aid] = text
            done += 1
            if done % 50 == 0:
                save_json("body_texts.json", cache)
                print(f"    {done}/{len(pending)} bodies fetched ({time.time() - t0:.1f}s)")
    save_json("body_texts.json", cache)
    print(f"  body fetch done in {time.time() - t0:.1f}s")
    return cache


# -------------------- Stage 5: slot extract (concurrent) --------------------

def _parse_lax(s: str) -> dict | None:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def extract_slots_concurrent(arxiv_ids: list[str], titles: dict[str, str], bodies: dict[str, str]) -> dict[str, dict]:
    cache = load_json("extracts.json", default={}) or {}
    # Skip retries for papers that have been visited (even if marked null) --
    # the gateway quota is exhausted; mark all unvisited as null and skip
    # ahead to stages 6-9 with whatever subset succeeded.
    skip_nulls = os.environ.get("TIER1_SKIP_NULL_RETRY") == "1"
    if skip_nulls:
        pending = [a for a in arxiv_ids if a not in cache]
        print(f"  slot extract (skip-null-retry): {len(cache)} cached, {len(pending)} pending")
    else:
        pending = [a for a in arxiv_ids if a not in cache or not cache[a]]
        print(f"  slot extract: {len(cache)} cached, {len(pending)} pending")

    def _extract(aid: str) -> tuple[str, dict | None]:
        body = bodies.get(aid, "")
        if not body or len(body) < 200:
            return aid, None
        prompt = EXTRACT_PROMPT.format(title=titles.get(aid, ""), body=body[:6000])
        try:
            # 2.5-flash has much higher rate-limit budget on the CF gateway
            # than 3.5-flash; 1000+ extraction calls would burn through 3.5's
            # quota in minutes. 2.5 is good enough for the structured extract
            # task -- quality matters more at the cheap-judge step.
            raw = gemini_generate(prompt, model="gemini-2.5-flash",
                                  json_mode=True, max_tokens=600, thinking_budget=0)
            return aid, _parse_lax(raw)
        except Exception as e:
            print(f"    extract {aid} err: {e}", file=sys.stderr)
            return aid, None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=EXTRACT_PARALLELISM) as pool:
        futs = {pool.submit(_extract, a): a for a in pending}
        done = 0
        for fut in as_completed(futs):
            aid, ext = fut.result()
            cache[aid] = ext
            done += 1
            if done % 100 == 0:
                save_json("extracts.json", cache)
                print(f"    {done}/{len(pending)} extracts ({time.time() - t0:.1f}s)")
    save_json("extracts.json", cache)
    print(f"  slot extract done in {time.time() - t0:.1f}s")
    return cache


# -------------------- Stage 6: slot embed (batched) -----------------------

def embed_slots(extracts: dict[str, dict]) -> dict[str, list[float]]:
    cache = load_json("slot_embs.json", default={}) or {}
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    for aid, ext in extracts.items():
        if aid in cache and cache[aid]:
            continue
        if not ext:
            cache[aid] = None
            continue
        text = slot_text(ext)
        if not text:
            cache[aid] = None
            continue
        pending_ids.append(aid)
        pending_texts.append(text)
    print(f"  slot embed: {len(cache) - sum(1 for v in cache.values() if v is None)} cached, {len(pending_texts)} pending")
    if pending_texts:
        t0 = time.time()
        vecs = gemini_embed_batch(pending_texts)
        for aid, v in zip(pending_ids, vecs):
            cache[aid] = v
        save_json("slot_embs.json", cache)
        print(f"  slot embed done in {time.time() - t0:.1f}s")
    return cache


# -------------------- Stage 7: slot-cosine rerank --------------------------

def slot_rerank(pairs: list[dict], slot_embs: dict[str, list[float]]) -> list[dict]:
    out = []
    for p in pairs:
        a, b = p["a"]["arxiv_id"], p["b"]["arxiv_id"]
        va, vb = slot_embs.get(a), slot_embs.get(b)
        if not va or not vb:
            continue
        ua = np.asarray(va, dtype=np.float32)
        ub = np.asarray(vb, dtype=np.float32)
        cos = float((ua @ ub) / (np.linalg.norm(ua) * np.linalg.norm(ub)))
        out.append({**p, "slot_cosine_dist": round(1 - cos, 6)})
    out.sort(key=lambda x: x["slot_cosine_dist"])
    out = out[:TOP_N_AFTER_SLOT]
    save_json("reranked_pairs.json", {"n_output": len(out), "pairs": out})
    print(f"  slot rerank: {len(out)} pairs after top-{TOP_N_AFTER_SLOT}")
    return out


# -------------------- Stage 8: cheap-judge (concurrent) --------------------

def cheap_judge(pairs: list[dict], extracts: dict[str, dict]) -> list[dict]:
    cache = load_json("cheap_judge.json", default={}) or {}

    def _judge(p: dict) -> tuple[str, dict | None]:
        key = f"{p['a']['arxiv_id']}__{p['b']['arxiv_id']}"
        if key in cache:
            return key, cache[key]
        ea = extracts.get(p["a"]["arxiv_id"]) or {}
        eb = extracts.get(p["b"]["arxiv_id"]) or {}
        prompt = JUDGE_PROMPT.format(
            a_arxiv=p["a"]["arxiv_id"], a_title=p["a"]["title"],
            a_problem=ea.get("problem", ""), a_methods=ea.get("methods", ""),
            b_arxiv=p["b"]["arxiv_id"], b_title=p["b"]["title"],
            b_problem=eb.get("problem", ""), b_methods=eb.get("methods", ""),
        )
        try:
            raw = gemini_generate(prompt, json_mode=True, max_tokens=200, thinking_budget=0)
            return key, _parse_lax(raw)
        except Exception as e:
            print(f"    judge {key} err: {e}", file=sys.stderr)
            return key, None

    pending = [p for p in pairs if f"{p['a']['arxiv_id']}__{p['b']['arxiv_id']}" not in cache]
    print(f"  cheap-judge: {len(cache)} cached, {len(pending)} pending")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=JUDGE_PARALLELISM) as pool:
        futs = {pool.submit(_judge, p): p for p in pending}
        done = 0
        for fut in as_completed(futs):
            key, verdict = fut.result()
            cache[key] = verdict
            done += 1
            if done % 50 == 0:
                save_json("cheap_judge.json", cache)
                print(f"    {done}/{len(pending)} judged ({time.time() - t0:.1f}s)")
    save_json("cheap_judge.json", cache)
    print(f"  cheap-judge done in {time.time() - t0:.1f}s")

    # Filter pairs to promising ones, preserving slot-cosine order.
    promising = []
    for p in pairs:
        key = f"{p['a']['arxiv_id']}__{p['b']['arxiv_id']}"
        v = cache.get(key)
        if v and v.get("verdict") == "promising":
            promising.append({**p, "judge": v})
    promising = promising[:TOP_M_AFTER_JUDGE]
    save_json("judged_pairs.json", {"n_promising": len(promising), "pairs": promising})
    print(f"  after cheap-judge: {len(promising)} promising pairs")
    return promising


# -------------------- Stage 9: Claude prompt prep -------------------------

def build_claude_prompts(pairs: list[dict], bodies: dict[str, str]) -> None:
    prompts = []
    for p in pairs:
        a, b = p["a"], p["b"]
        prompt = SUBAGENT_PROMPT.format(
            a_arxiv=a["arxiv_id"], a_sub=a.get("subfield", "?"), a_title=a["title"],
            a_body=bodies.get(a["arxiv_id"], "")[:6000],
            b_arxiv=b["arxiv_id"], b_sub=b.get("subfield", "?"), b_title=b["title"],
            b_body=bodies.get(b["arxiv_id"], "")[:6000],
            hamming=p["distance_norm"], body=p["slot_cosine_dist"],
        )
        prompts.append({
            "pair_key": f"{a['arxiv_id']}__{b['arxiv_id']}",
            "a_arxiv": a["arxiv_id"],
            "b_arxiv": b["arxiv_id"],
            "judge_mediator": p["judge"].get("mediator_guess"),
            "prompt": prompt,
        })
    save_json("tier1_bridge_prompts.json", prompts)
    print(f"  wrote {len(prompts)} Claude prompts -> data/tier1_bridge_prompts.json")


# -------------------- Main -------------------------------------------------

def main() -> None:
    meta = load_json("metadata.json")
    if not meta:
        print("ERROR: run tier1_assemble.py first", file=sys.stderr)
        sys.exit(1)

    pairs = band_scan_with_dedup()
    unique_ids = sorted({p["a"]["arxiv_id"] for p in pairs} | {p["b"]["arxiv_id"] for p in pairs})
    print(f"\nUnique papers across {len(pairs)} pairs: {len(unique_ids)}")

    titles = {m["arxiv_id"]: m["title"] for m in meta if m.get("arxiv_id")}

    print("\nStage 4: bodies")
    bodies = fetch_bodies_concurrent(unique_ids)

    print("\nStage 5: slot extract")
    extracts = extract_slots_concurrent(unique_ids, titles, bodies)

    print("\nStage 6: slot embed")
    slot_embs = embed_slots(extracts)

    print("\nStage 7: slot-cosine rerank")
    reranked = slot_rerank(pairs, slot_embs)

    print("\nStage 8: cheap-LLM-judge")
    promising = cheap_judge(reranked, extracts)

    print("\nStage 9: build Claude prompts")
    build_claude_prompts(promising, bodies)

    print(f"\nTier 1 pipeline complete. Top {len(promising)} pairs await Claude bridge-attempt.")


if __name__ == "__main__":
    main()
