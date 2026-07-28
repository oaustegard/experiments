"""Stage 5 — LLM bridge-attempt.

For each of the top 20 reranked pairs:
  1. 4-slot extraction from each paper (problem, methods, regime, generalization)
     using gemini-2.5-flash (light reasoning, fast)
  2. Bridge-attempt over the pair using gemini-2.5-flash with the
     mediator-concept prompt from phase-0; output is structured
     {compatibility: high/medium/low/none, rationale, sketch}.

The issue spec calls for gemini-flash-3 for extraction; we use the latest
available (gemini-2.5-flash) through the Cloudflare AI Gateway since that's
what's deployable here. Bridge-attempt was originally spec'd for Anthropic
Claude, but we don't have an Anthropic API key in this environment — using
Gemini 2.5-flash with extended thinking_budget for the bridge step instead.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gemini_generate, load_json, save_json  # noqa: E402

TOP_K = 20
BODY_FETCH_INTERVAL = 0.0  # bodies are cached from stage 4

EXTRACT_PROMPT = """You are extracting structured fields from a research paper.

Title: {title}
arXiv ID: {arxiv_id}
Body snippet (chars 2000–8000 of the full paper):
\"\"\"
{body}
\"\"\"

Return a JSON object with exactly these four keys:
- problem:               one short sentence stating the problem the paper solves
- methods:               one short sentence naming the main techniques used
- regime:                one short sentence stating the scale/setting where the result applies (e.g. "asymptotic in n", "small finite point sets", "constant-rate codes")
- hypothesized_generalization: one short sentence speculating about what generalization of the result the authors hint at (or "none" if they don't)

Output ONLY the JSON object, no preamble or markdown fences."""

BRIDGE_PROMPT = """You are assessing whether two research papers from different subfields share a deep structural connection that could form the basis of a "bridge" — a result that transfers ideas from one to the other.

Paper A — {a_arxiv}
Title: {a_title}
Extraction:
{a_extract}

Paper B — {b_arxiv}
Title: {b_title}
Extraction:
{b_extract}

Cross-field band info: SPECTER2 placed these in the cross-subfield band (normalized Hamming {hamming:.3f}); Gemini-body re-rank cosine {body:.3f}.

IMPORTANT — common failure mode to avoid: dismissing the pair because surface vocabulary differs or collides. Two papers may share a word with different meanings, OR may share deep structure with no shared vocabulary. Consider whether a MEDIATOR concept (e.g. Minkowski embedding, lattice constructions, algebraic curves giving discrete point sets, sphere packings, expander graphs, Galois cohomology, error-correcting codes from class field theory) could connect them. The bridge does not need to be famous; it needs to be plausible enough that a competent researcher could attempt the transfer.

Return a JSON object with these keys:
- compatibility: one of "high", "medium", "low", "none"
- rationale: 1–3 sentences explaining the scoring decision
- mediator: the bridge concept that might link them, or "none" if none identified
- sketch: 2–4 sentences outlining what a bridge result might look like — even when compatibility is low, sketch what a researcher *would* try if they had to bridge them. This is the most valuable output.

Output ONLY the JSON object, no preamble or markdown fences."""


def parse_json_lax(s: str) -> dict | None:
    """Strip markdown fences, locate the first JSON object, parse."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Find first { ... last }
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end < start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def extract_paper(arxiv_id: str, title: str, body: str) -> dict | None:
    """4-slot extraction for one paper."""
    prompt = EXTRACT_PROMPT.format(arxiv_id=arxiv_id, title=title, body=body[:6000])
    raw = gemini_generate(prompt, json_mode=True, max_tokens=600, thinking_budget=0)
    return parse_json_lax(raw)


def bridge_attempt(a: dict, b: dict, a_ext: dict, b_ext: dict, hamming: float, body: float) -> dict | None:
    """Bridge-attempt over a pair. Uses thinking budget for the harder reasoning."""
    prompt = BRIDGE_PROMPT.format(
        a_arxiv=a["arxiv_id"],
        a_title=a["title"],
        a_extract=json.dumps(a_ext, indent=2),
        b_arxiv=b["arxiv_id"],
        b_title=b["title"],
        b_extract=json.dumps(b_ext, indent=2),
        hamming=hamming,
        body=body,
    )
    # Give thinking budget for the bridge reasoning (this is the substantive step).
    raw = gemini_generate(prompt, json_mode=True, max_tokens=2000, thinking_budget=2048)
    return parse_json_lax(raw)


def main() -> None:
    rr = load_json("reranked_pairs.json")
    if not rr:
        print("ERROR: run stage4 first.", file=sys.stderr)
        sys.exit(1)
    bodies = load_json("body_embeddings.json", default={}) or {}
    # We need body TEXT not embeddings. Stage 4 didn't save text — re-fetch
    # from the cache by re-fetching arxiv. Cache body text alongside meta.
    body_text_cache_path = Path(__file__).resolve().parent.parent / "data" / "body_texts.json"
    body_texts: dict[str, str] = {}
    if body_text_cache_path.exists():
        body_texts = load_json("body_texts.json") or {}

    pairs = rr["pairs"][:TOP_K]
    print(f"Phase A stage 5: bridge-attempt over {len(pairs)} pairs")

    # Collect needed bodies.
    needed = sorted({p["a"]["arxiv_id"] for p in pairs} | {p["b"]["arxiv_id"] for p in pairs})
    missing_text = [a for a in needed if a not in body_texts or not body_texts[a]]
    if missing_text:
        print(f"  fetching body text for {len(missing_text)} papers (not cached)…")
        from stage4_rerank import fetch_body, body_window
        for i, arxiv_id in enumerate(missing_text, 1):
            t = body_window(fetch_body(arxiv_id))
            body_texts[arxiv_id] = t
            print(f"    [{i}/{len(missing_text)}] {arxiv_id}: {len(t)} chars")
            time.sleep(0.5)
        save_json("body_texts.json", body_texts)

    # --- Extract ----------------------------------------------------------
    extractions = load_json("extractions.json", default={}) or {}
    for i, arxiv_id in enumerate(needed, 1):
        if arxiv_id in extractions:
            continue
        body = body_texts.get(arxiv_id, "")
        title = next((p["a"]["title"] if p["a"]["arxiv_id"] == arxiv_id else p["b"]["title"])
                     for p in pairs if arxiv_id in (p["a"]["arxiv_id"], p["b"]["arxiv_id"]))
        if not body:
            extractions[arxiv_id] = None
            print(f"  extract [{i}/{len(needed)}] {arxiv_id}: no body, skipping")
            continue
        ext = extract_paper(arxiv_id, title, body)
        extractions[arxiv_id] = ext
        ok = ext is not None
        print(f"  extract [{i}/{len(needed)}] {arxiv_id}: {'ok' if ok else 'parse-failed'}")
        if i % 5 == 0:
            save_json("extractions.json", extractions)
    save_json("extractions.json", extractions)

    # --- Bridge-attempt ---------------------------------------------------
    attempts = load_json("bridge_attempts.json", default={"results": []}) or {"results": []}
    done_keys = {(r["a"]["arxiv_id"], r["b"]["arxiv_id"]) for r in attempts["results"]}
    results = list(attempts["results"])

    for idx, p in enumerate(pairs, 1):
        key = (p["a"]["arxiv_id"], p["b"]["arxiv_id"])
        if key in done_keys:
            continue
        a_ext = extractions.get(p["a"]["arxiv_id"])
        b_ext = extractions.get(p["b"]["arxiv_id"])
        if not a_ext or not b_ext:
            print(f"  bridge [{idx}/{len(pairs)}] skipping {key} — missing extraction")
            continue
        print(f"  bridge [{idx}/{len(pairs)}] {key} [{p['kind']}]…")
        t0 = time.time()
        bridge = bridge_attempt(
            p["a"], p["b"], a_ext, b_ext,
            p["distance_norm"], p["gemini_body_cosine_dist"],
        )
        ok = bridge is not None
        comp = (bridge or {}).get("compatibility", "?")
        print(f"    -> {'ok' if ok else 'parse-failed'} compatibility={comp}  ({time.time() - t0:.1f}s)")
        results.append({
            "a": p["a"],
            "b": p["b"],
            "kind": p["kind"],
            "distance_norm": p["distance_norm"],
            "gemini_body_cosine_dist": p["gemini_body_cosine_dist"],
            "a_extract": a_ext,
            "b_extract": b_ext,
            "bridge": bridge,
        })
        # checkpoint every result
        save_json("bridge_attempts.json", {"results": results})

    save_json("bridge_attempts.json", {"results": results})

    n_sketches = sum(1 for r in results if r.get("bridge") and (r["bridge"].get("sketch") or "").strip())
    print(f"\nStage 5 complete:")
    print(f"  total bridge attempts: {len(results)}")
    print(f"  non-empty sketches:    {n_sketches}")
    by_comp: dict[str, int] = {}
    for r in results:
        c = (r.get("bridge") or {}).get("compatibility", "?")
        by_comp[c] = by_comp.get(c, 0) + 1
    print(f"  compatibility distribution: {by_comp}")


if __name__ == "__main__":
    main()
