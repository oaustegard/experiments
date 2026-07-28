"""Stage 9 — Claude subagent formal translation.

For each "resolves" / "partially_resolves" candidate pair from te_judged.json,
produces a 3-5 sentence translation:
  - which empirical quantity corresponds to which theory variable
  - which empirical regime maps to which theoretical hypothesis
  - what predictions does T make about E that the authors haven't tested

This script is designed to be called from an orchestrator session that
holds the Agent tool. It can also run in standalone mode via Anthropic
SDK (ANTHROPIC_API_KEY required).

Output:
  data/te_translations.json — [{emp_arxiv, th_arxiv, translation, ...}, ...]

Usage (standalone):
  python te_translate.py [--max-pairs 30] [--model claude-opus-4-7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import ANTHROPIC_API_KEY, load_json, save_json  # noqa: E402

CHECKPOINT_EVERY = 5

TRANSLATION_PROMPT = """You are writing a technical translation between an empirical ML/CS observation and a theoretical mathematics result.

Empirical paper {emp_arxiv}: "{emp_title}"
  Phenomenon: {phenomenon}
  Regime: {emp_regime}
  What the authors cannot explain: {mechanism_unknown}

Theory paper {th_arxiv}: "{th_title}"
  Theorem: {theorem_claim}
  Regime: {th_regime}
  Technique: {mechanism_provided}

Judge summary: {judge_translation}

Write a 3-5 sentence technical translation for an applied-ML researcher:

1. State which empirical quantity corresponds to which variable in the theorem.
2. State which empirical regime maps to which theoretical hypothesis (note any gaps or required assumptions).
3. State what specific, testable prediction the theorem makes about the empirical phenomenon that the authors have not tested.
4. If the correspondence is imperfect, state the strongest version of the connection that can be made.

Write the translation directly — no preamble, no meta-commentary. Use concrete variable names where possible."""


def translate_via_sdk(pair: dict) -> str | None:
    """Translate using Anthropic SDK directly (standalone mode)."""
    if not ANTHROPIC_API_KEY:
        print("  ANTHROPIC_API_KEY not set — cannot translate", file=sys.stderr)
        return None
    try:
        import anthropic
    except ImportError:
        print("  anthropic SDK not installed", file=sys.stderr)
        return None

    emp_slots = pair.get("emp_slots") or {}
    th_slots  = pair.get("th_slots")  or {}
    prompt = TRANSLATION_PROMPT.format(
        emp_arxiv=pair["emp_arxiv"],
        th_arxiv=pair["th_arxiv"],
        emp_title=pair.get("emp_title", ""),
        th_title=pair.get("th_title", ""),
        phenomenon=emp_slots.get("phenomenon", "(missing)"),
        emp_regime=emp_slots.get("regime", "(missing)"),
        mechanism_unknown=emp_slots.get("mechanism_unknown", "(missing)"),
        theorem_claim=th_slots.get("theorem_claim", "(missing)"),
        th_regime=th_slots.get("regime", "(missing)"),
        mechanism_provided=th_slots.get("mechanism_provided", "(missing)"),
        judge_translation=pair.get("judge_translation", "(none)"),
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=os.environ.get("TE_TRANSLATE_MODEL", "claude-opus-4-7"),
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip() if resp.content else None


def build_subagent_prompts(pairs: list[dict]) -> list[dict]:
    """Return prompts for orchestrator Agent tool use (non-standalone mode)."""
    out = []
    for p in pairs:
        emp_slots = p.get("emp_slots") or {}
        th_slots  = p.get("th_slots")  or {}
        prompt = TRANSLATION_PROMPT.format(
            emp_arxiv=p["emp_arxiv"],
            th_arxiv=p["th_arxiv"],
            emp_title=p.get("emp_title", ""),
            th_title=p.get("th_title", ""),
            phenomenon=emp_slots.get("phenomenon", "(missing)"),
            emp_regime=emp_slots.get("regime", "(missing)"),
            mechanism_unknown=emp_slots.get("mechanism_unknown", "(missing)"),
            theorem_claim=th_slots.get("theorem_claim", "(missing)"),
            th_regime=th_slots.get("regime", "(missing)"),
            mechanism_provided=th_slots.get("mechanism_provided", "(missing)"),
            judge_translation=p.get("judge_translation", "(none)"),
        )
        out.append({
            "key": f'{p["emp_arxiv"]}:{p["th_arxiv"]}',
            "emp_arxiv": p["emp_arxiv"],
            "th_arxiv": p["th_arxiv"],
            "prompt": prompt,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=30)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--emit-prompts", action="store_true",
                    help="Print subagent prompts JSON instead of calling SDK")
    args = ap.parse_args()

    os.environ.setdefault("TE_TRANSLATE_MODEL", args.model)

    judged = load_json("te_judged.json", default={})
    if not judged:
        print("ERROR: run te_judge.py first", file=sys.stderr)
        sys.exit(1)

    candidates = [
        p for p in (judged.get("pairs") or [])
        if p.get("judge_verdict") in ("resolves", "partially_resolves")
    ]
    # Sort by slot cosine descending
    candidates.sort(key=lambda p: p.get("slot_cosine", 0.0), reverse=True)
    candidates = candidates[: args.max_pairs]
    print(f"Translation targets: {len(candidates)} pairs (max={args.max_pairs})")

    if args.emit_prompts:
        prompts = build_subagent_prompts(candidates)
        print(json.dumps(prompts, indent=2))
        return

    # Standalone SDK mode
    existing = {
        f'{t["emp_arxiv"]}:{t["th_arxiv"]}': t
        for t in (load_json("te_translations.json", default={}).get("translations") or [])
        if t.get("translation")
    }
    need = [p for p in candidates if f'{p["emp_arxiv"]}:{p["th_arxiv"]}' not in existing]
    print(f"  {len(existing)} cached, {len(need)} to translate")

    for idx, pair in enumerate(need, 1):
        key = f'{pair["emp_arxiv"]}:{pair["th_arxiv"]}'
        print(f"  [{idx}/{len(need)}] {key}…")
        translation = translate_via_sdk(pair)
        existing[key] = {
            "emp_arxiv": pair["emp_arxiv"],
            "th_arxiv":  pair["th_arxiv"],
            "emp_title": pair.get("emp_title", ""),
            "th_title":  pair.get("th_title", ""),
            "judge_verdict": pair.get("judge_verdict", ""),
            "slot_cosine":   pair.get("slot_cosine", 0.0),
            "translation": translation,
        }
        if idx % CHECKPOINT_EVERY == 0:
            save_json("te_translations.json", {
                "n": len(existing),
                "translations": list(existing.values()),
            })
        time.sleep(0.5)

    save_json("te_translations.json", {
        "n": len(existing),
        "translations": list(existing.values()),
    })
    ok = sum(1 for t in existing.values() if t.get("translation"))
    print(f"\nDone: {ok}/{len(existing)} translations written")


if __name__ == "__main__":
    main()
