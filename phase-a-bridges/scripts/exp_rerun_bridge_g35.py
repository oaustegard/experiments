"""Experiment 1 — Re-run run 1's bridge step with gemini-3.5-flash.

Same 20 pairs, same prompts as the original stage 5 — only the model
changes. Writes to bridge_attempts_g35.json so we can diff against the
existing gemini-2.5 bridge_attempts.json and the Claude-subagent
bridge_attempts_claude.json from run 2.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gemini_generate, load_json, save_json  # noqa: E402
from stage5_bridge import (  # noqa: E402
    BRIDGE_PROMPT,
    EXTRACT_PROMPT,
    parse_json_lax,
)


MODEL = "gemini-3.5-flash"
TOP_K = 20


def extract_paper(arxiv_id: str, title: str, body: str) -> dict | None:
    prompt = EXTRACT_PROMPT.format(arxiv_id=arxiv_id, title=title, body=body[:6000])
    raw = gemini_generate(prompt, model=MODEL, json_mode=True, max_tokens=600, thinking_budget=0)
    return parse_json_lax(raw)


def bridge(a: dict, b: dict, a_ext: dict, b_ext: dict, hamming: float, body_cos: float) -> dict | None:
    import json as _json
    prompt = BRIDGE_PROMPT.format(
        a_arxiv=a["arxiv_id"], a_title=a["title"], a_extract=_json.dumps(a_ext, indent=2),
        b_arxiv=b["arxiv_id"], b_title=b["title"], b_extract=_json.dumps(b_ext, indent=2),
        hamming=hamming, body=body_cos,
    )
    raw = gemini_generate(prompt, model=MODEL, json_mode=True, max_tokens=2000, thinking_budget=2048)
    return parse_json_lax(raw)


def main() -> None:
    rr = load_json("reranked_pairs.json")
    if not rr:
        print("ERROR: reranked_pairs.json missing.", file=sys.stderr)
        sys.exit(1)
    bodies = load_json("body_texts.json", default={}) or {}
    pairs = rr["pairs"][:TOP_K]
    print(f"Exp 1: re-running bridge step on {len(pairs)} pairs with {MODEL}")

    extractions = {}
    needed = sorted({p["a"]["arxiv_id"] for p in pairs} | {p["b"]["arxiv_id"] for p in pairs})
    for i, arxiv_id in enumerate(needed, 1):
        body = bodies.get(arxiv_id, "")
        title = next(
            (p["a"]["title"] if p["a"]["arxiv_id"] == arxiv_id else p["b"]["title"])
            for p in pairs
            if arxiv_id in (p["a"]["arxiv_id"], p["b"]["arxiv_id"])
        )
        if not body:
            extractions[arxiv_id] = None
            print(f"  extract [{i}/{len(needed)}] {arxiv_id}: no body")
            continue
        ext = extract_paper(arxiv_id, title, body)
        extractions[arxiv_id] = ext
        print(f"  extract [{i}/{len(needed)}] {arxiv_id}: {'ok' if ext else 'parse-fail'}")
    save_json("extractions_g35.json", extractions)

    results = []
    for idx, p in enumerate(pairs, 1):
        a_ext = extractions.get(p["a"]["arxiv_id"])
        b_ext = extractions.get(p["b"]["arxiv_id"])
        if not a_ext or not b_ext:
            print(f"  bridge [{idx}/{len(pairs)}] skipped — missing extraction")
            continue
        t0 = time.time()
        out = bridge(p["a"], p["b"], a_ext, b_ext, p["distance_norm"], p["gemini_body_cosine_dist"])
        comp = (out or {}).get("compatibility", "?")
        print(f"  bridge [{idx}/{len(pairs)}] {p['a']['arxiv_id']} ↔ {p['b']['arxiv_id']} -> {comp}  ({time.time() - t0:.1f}s)")
        results.append({
            "a": p["a"], "b": p["b"], "kind": p["kind"],
            "distance_norm": p["distance_norm"],
            "gemini_body_cosine_dist": p["gemini_body_cosine_dist"],
            "a_extract": a_ext, "b_extract": b_ext,
            "bridge": out,
        })
        save_json("bridge_attempts_g35.json", {"model": MODEL, "results": results})

    save_json("bridge_attempts_g35.json", {"model": MODEL, "results": results})
    by_comp: dict[str, int] = {}
    for r in results:
        c = (r.get("bridge") or {}).get("compatibility", "?")
        by_comp[c] = by_comp.get(c, 0) + 1
    print(f"\nDone: {len(results)} attempts. Compatibility distribution: {by_comp}")


if __name__ == "__main__":
    main()
