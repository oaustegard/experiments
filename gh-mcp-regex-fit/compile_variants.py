#!/usr/bin/env python3
"""Three ways to let a model author the rules, and the supervision each one gets.

`gemini_arms.compile_rules` is the clean room: the model sees the 79 targets and
the 23 cue names, nothing else. This module adds the two variants that give it
supervision, so the question "does *fitting* fail, or did Claude's greedy
covering algorithm fail" gets an answer that is not confounded by the fitter.

* **fitter**   — the clean-room prompt plus labelled family-A rows. The honest
                 competitor to `fit.py`'s decision-list induction: same
                 supervision, different hypothesis-class-author.
* **iterated** — clean-room rules, then its own family-A errors fed back for
                 revision. This is what "trained" most naturally means for a
                 model, and it is the only arm that sees a loss signal.

Both are scored on family B and wild only. Family A is in-sample for them and
its number is reported for completeness, not comparison.

    python3 compile_variants.py fitter   --model gemini-3.7-flash --per-label 3
    python3 compile_variants.py iterated --model gemini-3.7-flash --rounds 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from arms import labels
from eval import load_split, score
from gemini_arms import CompiledRouter, build_prompt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SUPERVISED_SUFFIX = """

## Labelled examples

Here are real requests with the target each one should route to. They are drawn \
from one phrasing family; requests in the wild will be phrased differently, so \
generalise from these rather than matching them literally.

{examples}

Write rules that would route these correctly AND would still route a paraphrase \
correctly. Output ONLY the JSON array."""

REVISE_SUFFIX = """

## Your previous rules

{rules}

## Where they went wrong

Each line is a request, the target it should have routed to, and what your rules \
actually produced (`null` means no rule matched and the router abstained).

{errors}

Revise the rule list. Keep what works, fix what does not. Watch rule *order*: a \
rule that fires too early blocks every rule after it, which is the usual cause \
of a wrong label where a later rule would have been right. Abstentions mean a \
missing rule or a precondition that is too strict.

Output ONLY the revised JSON array, complete — it replaces the previous list."""


def _sample(path: Path, per_label: int, seed: int) -> list[dict]:
    rows = [r for r in load_split(path) if r.get("label")]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    out = []
    for lab in sorted(by):
        out += rng.sample(by[lab], min(per_label, len(by[lab])))
    return out


def _write(rules: list, model: str, tag: str, extra: dict) -> Path:
    valid = set(labels())
    kept = [r for r in rules if isinstance(r, dict) and r.get("label") in valid]
    ok = []
    for r in kept:
        try:
            re.compile(r.get("pattern") or "", re.I)
            ok.append(r)
        except re.error:
            pass
    out = HERE / f"rules_{tag}.json"
    out.write_text(json.dumps({
        "author": model, "n_returned": len(rules), "n_kept": len(ok),
        "invented_labels": sorted({r.get("label") for r in rules
                                   if isinstance(r, dict) and r.get("label") not in valid}),
        **extra, "rules": ok}, indent=1) + "\n")
    print(f"{tag}: {len(rules)} returned, {len(ok)} valid -> {out.name}")
    return out


def _ask(prompt: str, model: str) -> list:
    from gemini_client import generate
    txt = generate(prompt, model=model, thinking_budget=-1,
                   max_output_tokens=32768, response_json=True)
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    obj = json.loads(txt)
    return obj if isinstance(obj, list) else obj.get("rules", [])


def compile_fitter(model: str, per_label: int, seed: int, tag: str | None) -> Path:
    ex = _sample(DATA / "family_a.jsonl", per_label, seed)
    lines = "\n".join(f'- "{r["query"]}"  ->  {r["label"]}' for r in ex)
    prompt = build_prompt() + SUPERVISED_SUFFIX.format(examples=lines)
    return _write(_ask(prompt, model), model, tag or f"{model}-fitter",
                  {"supervision": f"{len(ex)} labelled family-A rows",
                   "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                   "prompt_chars": len(prompt)})


def compile_iterated(model: str, rounds: int, seed: int, tag: str | None,
                     start: Path | None, max_errors: int = 120) -> Path:
    cur = json.loads((start or HERE / "rules_gemini-cleanroom.json").read_text())["rules"]
    rows = [r for r in load_split(DATA / "family_a.jsonl") if r.get("label")]
    rng = random.Random(seed)
    history = []
    path = start or HERE / "rules_gemini-cleanroom.json"
    for i in range(1, rounds + 1):
        s = score(CompiledRouter(path), rows)
        errs = s["errors"]
        history.append({"round": i - 1, "family_a_acc": s["label_acc"], "n_errors": len(errs)})
        print(f"  round {i}: starting from family-A acc {s['label_acc']:.3f}, {len(errs)} errors")
        shown = rng.sample(errs, min(max_errors, len(errs)))
        elines = "\n".join(
            f'- "{e["query"]}"  should be {e["gold"]}  but got {e["got"]}' for e in shown)
        prompt = build_prompt() + REVISE_SUFFIX.format(
            rules=json.dumps(cur, indent=0), errors=elines)
        cur = _ask(prompt, model)
        path = _write(cur, model, f"{tag or model}-iter{i}",
                      {"supervision": f"round {i}; {len(shown)} family-A errors shown",
                       "history": history})
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fitter", "iterated"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-label", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--start", type=Path, default=None)
    a = ap.parse_args()
    from gemini_client import DEFAULT_MODEL
    model = a.model or DEFAULT_MODEL
    if a.cmd == "fitter":
        compile_fitter(model, a.per_label, a.seed, a.tag)
    else:
        compile_iterated(model, a.rounds, a.seed, a.tag, a.start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
