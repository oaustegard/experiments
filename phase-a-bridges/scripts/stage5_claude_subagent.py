"""Stage 5 orchestration helper — Claude subagent variant.

Used by the main session (which holds the Agent tool) rather than invoked
as a subprocess. Provides:

  - build_bridge_prompts() — produces a JSON list of {pair_key, prompt}
    items that the orchestrator can hand to N parallel Agent calls
  - save_subagent_result(pair_key, agent_text) — parses the agent's
    JSON output and merges it into bridge_attempts_claude.json

The agent prompt includes both extraction (4-slot) and bridge-attempt
in a single shot, so each pair is one subagent invocation. This is the
"can't you just spawn sub agents from here?" path that replaces the
Gemini-2.5-flash fallback used when no Anthropic API key is exposed
inside the container.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_json, save_json  # noqa: E402


SUBAGENT_PROMPT = """You are assessing whether two research papers from different subfields share a deep structural connection that could form the basis of a "bridge" — a result that transfers ideas from one to the other.

Do BOTH steps in one response:

# STEP 1 — Extract a 4-slot record for each paper from its body snippet.

# STEP 2 — Assess the bridge.

IMPORTANT — common failure mode to avoid: dismissing the pair because surface vocabulary differs or collides. Two papers may share a word with different meanings, OR may share deep structure with no shared vocabulary. Consider whether a MEDIATOR concept (e.g. Minkowski embedding, lattice constructions, algebraic curves giving discrete point sets, sphere packings, expander graphs, Galois cohomology, error-correcting codes from class field theory) could connect them. The bridge does not need to be famous; it needs to be plausible enough that a competent researcher could attempt the transfer.

---

Paper A — {a_arxiv}  (subfield: {a_sub})
Title: {a_title}
Body snippet (chars 2000–8000 of the paper):
\"\"\"
{a_body}
\"\"\"

Paper B — {b_arxiv}  (subfield: {b_sub})
Title: {b_title}
Body snippet:
\"\"\"
{b_body}
\"\"\"

Cross-field band info: SPECTER2 normalized Hamming {hamming:.3f}; Gemini-body re-rank cosine distance {body:.3f}.

---

Return ONE JSON object with this exact shape (no preamble, no markdown fences, just the JSON):

{{
  "a_extract": {{
    "problem": "...",
    "methods": "...",
    "regime": "...",
    "hypothesized_generalization": "..."
  }},
  "b_extract": {{ ... same four keys ... }},
  "bridge": {{
    "compatibility": "high" | "medium" | "low" | "none",
    "rationale": "1-3 sentences",
    "mediator": "name of the bridge concept, or 'none'",
    "sketch": "2-4 sentences outlining what a bridge result might look like. Sketch what a researcher would try even when compatibility is low; this is the most valuable output."
  }}
}}
"""


def build_bridge_prompts(top_k: int = 20) -> list[dict]:
    rr = load_json("reranked_pairs.json")
    if not rr:
        print("ERROR: reranked_pairs.json missing", file=sys.stderr)
        return []
    bodies = load_json("body_texts.json", default={}) or {}
    pairs = rr["pairs"][:top_k]
    out: list[dict] = []
    for p in pairs:
        a, b = p["a"], p["b"]
        a_body = bodies.get(a["arxiv_id"], "")
        b_body = bodies.get(b["arxiv_id"], "")
        if not a_body or not b_body:
            out.append({
                "pair_key": f"{a['arxiv_id']}__{b['arxiv_id']}",
                "skipped": True,
                "reason": f"missing body text (a={len(a_body)} b={len(b_body)})",
                "pair": p,
            })
            continue
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
            body=p["gemini_body_cosine_dist"],
        )
        out.append({
            "pair_key": f"{a['arxiv_id']}__{b['arxiv_id']}",
            "skipped": False,
            "prompt": prompt,
            "pair": p,
        })
    return out


def parse_subagent_output(s: str) -> dict | None:
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


def save_results(results: list[dict]) -> None:
    save_json("bridge_attempts_claude.json", {"results": results})


if __name__ == "__main__":
    # Print prompts as JSON for the orchestrator's consumption.
    prompts = build_bridge_prompts()
    print(json.dumps([{"pair_key": p["pair_key"], "skipped": p.get("skipped", False)} for p in prompts], indent=2))
    print(f"\nTotal: {len(prompts)} prompts, {sum(1 for p in prompts if p.get('skipped'))} skipped", file=sys.stderr)
    # Also dump full prompts to a file so the orchestrator can read them.
    out_path = DATA / "bridge_prompts.json"
    out_path.write_text(json.dumps(prompts, indent=2))
    print(f"Wrote {out_path}", file=sys.stderr)
