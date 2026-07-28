"""Stage 8 — Binary cheap-LLM-judge via gemini-2.5-flash.

For each candidate (empirical, theory) pair in te_reranked.json, asks:
  Does theorem T, properly translated, resolve observation E in the
  regime where both apply?

Verdict: "resolves" | "partially_resolves" | "unrelated"

Output:
  data/te_judged.json — pairs annotated with judge_verdict + translation_hint

Resumable: reads existing te_judged.json and skips already-judged pairs.

Usage:
  python te_judge.py [--top-n 300] [--parallelism 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.__str__())
from te_common import gemini_generate, load_json, save_json  # noqa: E402

CHECKPOINT_EVERY = 20

JUDGE_PROMPT = """You are evaluating whether a theoretical mathematics result explains an empirical ML/CS observation.

Empirical paper ({emp_arxiv}):
Title: {emp_title}
Observation (phenomenon): {phenomenon}
Regime: {emp_regime}
What the authors admit they cannot explain: {mechanism_unknown}

Theory paper ({th_arxiv}):
Title: {th_title}
Result (theorem_claim): {theorem_claim}
Regime: {th_regime}
How it is proven: {mechanism_provided}

Question: Does the theorem in the theory paper, when properly translated to the empirical domain, RESOLVE or EXPLAIN the observation in the empirical paper in the regime where both apply?

Specifically:
- Does the theorem's claim (when abstracted) match the structure of the empirical observation?
- Do the regimes plausibly align (not necessarily identical, but compatible)?
- Would knowing this theorem help an ML researcher understand WHY the observation holds?

Answer with ONE JSON object:
{{
  "verdict": "resolves" | "partially_resolves" | "unrelated",
  "translation_hint": "One sentence: what the theorem variable corresponds to in the ML context, and what the theorem predicts. Or 'none' if unrelated.",
  "regime_alignment": "compatible" | "misaligned" | "unclear"
}}

Be strict — "resolves" requires the theorem to directly address the mechanistic gap in the empirical paper; "partially_resolves" means partial overlap; "unrelated" means no plausible connection at the claimed level.

Output ONLY the JSON object."""


def judge_pair(pair: dict) -> dict | None:
    emp_slots = pair.get("emp_slots") or {}
    th_slots  = pair.get("th_slots")  or {}
    prompt = JUDGE_PROMPT.format(
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
    )
    raw = gemini_generate(prompt, model="gemini-2.5-flash", json_mode=True,
                          max_tokens=400, thinking_budget=0)
    s = raw.strip().lstrip("`").lstrip("json").lstrip()
    s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n",      type=int, default=300)
    ap.add_argument("--parallelism",type=int, default=4)
    args = ap.parse_args()

    reranked = load_json("te_reranked.json")
    if not reranked:
        print("ERROR: run te_rerank.py first", file=sys.stderr)
        sys.exit(1)
    pairs = reranked["pairs"][: args.top_n]

    judged = {
        f'{p["emp_arxiv"]}:{p["th_arxiv"]}': p
        for p in (load_json("te_judged.json", default={}).get("pairs") or [])
        if p.get("judge_verdict")
    }
    need = [p for p in pairs if f'{p["emp_arxiv"]}:{p["th_arxiv"]}' not in judged]
    print(f"Judging {len(need)} pairs ({len(judged)} cached, top-{args.top_n} total)…")

    def _judge(p: dict) -> tuple[str, dict | None]:
        key = f'{p["emp_arxiv"]}:{p["th_arxiv"]}'
        result = judge_pair(p)
        return key, result

    done = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as ex:
        futs = {ex.submit(_judge, p): p for p in need}
        for fut in as_completed(futs):
            p = futs[fut]
            key = f'{p["emp_arxiv"]}:{p["th_arxiv"]}'
            try:
                _, result = fut.result()
            except Exception as e:
                print(f"  judge {key}: error {e}", file=sys.stderr)
                result = None
            annotated = dict(p)
            if result:
                annotated["judge_verdict"]      = result.get("verdict", "unrelated")
                annotated["judge_translation"]  = result.get("translation_hint", "")
                annotated["judge_regime_align"] = result.get("regime_alignment", "unclear")
            else:
                annotated["judge_verdict"] = "error"
            judged[key] = annotated
            done += 1
            verdict = annotated.get("judge_verdict", "?")
            print(f"  [{done}/{len(need)}] {p['emp_arxiv']} × {p['th_arxiv']}: {verdict}")
            if done % CHECKPOINT_EVERY == 0:
                save_json("te_judged.json", {"n_pairs": len(judged), "pairs": list(judged.values())})

    save_json("te_judged.json", {"n_pairs": len(judged), "pairs": list(judged.values())})

    resolves = [p for p in judged.values() if p.get("judge_verdict") in ("resolves", "partially_resolves")]
    print(f"\nDone: {len(resolves)}/{len(judged)} resolve/partially_resolve")


if __name__ == "__main__":
    main()
