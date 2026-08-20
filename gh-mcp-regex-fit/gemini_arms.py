#!/usr/bin/env python3
"""An independent model as the rule compiler, and as a live router.

The first pass called `handwritten.py` the "hand-written" arm and treated it as
a human baseline. It is not: those rules were written by Claude reading the 50
schemas, so the experiment's real contrast was *model reasoning compiled once
into deterministic rules* versus *statistics fitted from a corpus* — and it was
contaminated, because the same author wrote the query templates it was scored on.

Gemini fixes both problems at once. It did not write the eval, and it has never
seen it.

Two arms:

* **`gemini-compile`** — offline. The model gets the catalogue and the structural
  cue vocabulary, nothing else: no queries, no eval rows, no failure list, and
  none of Claude's rules. It returns ordered rules, which then run through the
  *same executor* as `handwritten.py`, so the comparison isolates authorship.
  This is what Oskar meant by a "trained" regex: a model reasons about the use
  case once, and the artefact is deterministic code at 0.04 ms.
* **`gemini-live`** — per query, at inference time. The ceiling the compiled
  arm is trying to approximate, and the escalation target a cascade would use.

    python3 gemini_arms.py compile        # author rules (needs the gateway)
    python3 gemini_arms.py eval           # score whatever rule files exist
    python3 gemini_arms.py live --split wild
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from arms import ArmBase, labels, register
from catalogue import load as load_catalogue
from cues import CUE_NAMES, cues, extract

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Executor — identical semantics to handwritten.py, so only authorship differs
# --------------------------------------------------------------------------
class CompiledRouter(ArmBase):
    """Ordered rules, first match wins, structural precondition, no fallback."""

    def __init__(self, rules_path: Path):
        blob = json.loads(Path(rules_path).read_text())
        self.meta = {k: v for k, v in blob.items() if k != "rules"}
        self.valid = set(labels())
        self.rules = []
        for r in blob["rules"]:
            if r["label"] not in self.valid:
                continue  # a target the model invented; dropped, and counted below
            pat = r.get("pattern") or ""
            try:
                rx = re.compile(pat, re.I) if pat else None
            except re.error:
                continue
            need = r.get("requires") or None
            if need and need not in CUE_NAMES:
                need = None
            self.rules.append((r["label"], rx, need))

    def route(self, query: str) -> str | None:
        c = cues(query)
        for label, rx, need in self.rules:
            if need and not c.get(need):
                continue
            if rx and not rx.search(query):
                continue
            return label
        return None


# --------------------------------------------------------------------------
# Offline compilation
# --------------------------------------------------------------------------
def render_catalogue() -> str:
    cat = load_catalogue("session")
    out = []
    for name, tool in sorted(cat.items()):
        req = ", ".join(tool["required"]) or "none"
        m = tool["params"].get("method")
        if m and m.get("enum"):
            out.append(f"- {name} (required: {req}) — {tool['description']}")
            gloss = m.get("description", "")
            for e in m["enum"]:
                out.append(f"    * {name}::{e}")
            # Four of the seven dispatchers gloss nothing ("The method to execute",
            # "The action to perform"); those lines are padding, not signal.
            if gloss and not re.fullmatch(r"The (method to execute|action to perform)", gloss):
                out.append(f"    method notes: {gloss}")
        else:
            out.append(f"- {name} (required: {req}) — {tool['description']}")
    return "\n".join(out)


COMPILE_PROMPT = """You are compiling a deterministic router for a GitHub tool catalogue.

A user types a natural-language request in a coding assistant. Your job is to \
write ordered regular-expression rules that map that request to exactly one of \
the routing targets below, or decline.

## The targets ({n_targets} of them)

{catalogue}

## Structural cues available to you

A separate extraction layer has already parsed the request and can tell you \
whether each of these is present. Use them as preconditions so that a keyword \
alone cannot fire a rule that needs a structural referent:

{cue_list}

## How the rules are executed

Rules are tried in order. The first rule whose `requires` cue is present (if it \
declares one) AND whose `pattern` matches the request wins. If no rule matches, \
the router ABSTAINS — this is correct and desirable for requests no tool serves. \
There is no fallback rule and you must not write one: a catch-all destroys the \
ability to decline.

## What to produce

A JSON array. Each element:

  {{"label": "<exact target from the list>",
    "pattern": "<Python regex, case-insensitive, or \\"\\" to match anything>",
    "requires": "<one cue name from the list, or null>"}}

Rules to follow:
- Order matters. Put specific rules before general ones.
- Enumerate synonyms people actually use. A request for a diff might say diff, \
patch, changeset, "what changed", "what code does this change".
- Prefer a structural precondition wherever the target needs a referent.
- Cover the targets you believe requests will actually hit. Omitting a target is \
better than a rule you do not believe in, because a wrong rule earlier in the \
order blocks every rule after it.
- Grammatical number is meaningful here: plural usually names a list endpoint, \
singular a fetch-one.

Output ONLY the JSON array. No prose, no markdown fence."""


def build_prompt() -> str:
    return COMPILE_PROMPT.format(
        n_targets=len(labels()),
        catalogue=render_catalogue(),
        cue_list="\n".join(f"- {c}" for c in CUE_NAMES),
    )


def compile_rules(model: str, tag: str | None = None) -> Path:
    """Ask the model for rules. Thinking is ON here: this is reasoning, not extraction."""
    from gemini_client import generate

    text = generate(build_prompt(), model=model, thinking_budget=-1,
                    max_output_tokens=32768, response_json=True)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    rules = json.loads(text)
    valid = set(labels())
    kept = [r for r in rules if isinstance(r, dict) and r.get("label") in valid]
    out = HERE / f"rules_{tag or model.replace('.', '')}.json"
    out.write_text(json.dumps({
        "author": model, "prompt_sha": __import__("hashlib").sha256(
            build_prompt().encode()).hexdigest()[:16],
        "n_returned": len(rules), "n_kept": len(kept),
        "invented_labels": sorted({r.get("label") for r in rules
                                   if isinstance(r, dict) and r.get("label") not in valid}),
        "rules": kept}, indent=1) + "\n")
    print(f"{model}: {len(rules)} rules returned, {len(kept)} valid, wrote {out.name}")
    return out


# --------------------------------------------------------------------------
# Live routing
# --------------------------------------------------------------------------
LIVE_PROMPT = """Route this request to exactly one target, or to null if no tool serves it.

Targets:
{catalogue}

Request: {query}

Reply with only JSON: {{"label": "<target>"}} or {{"label": null}}"""


class GeminiLiveArm(ArmBase):
    """Per-query routing. thinking_budget=0: this is extraction, not reasoning."""

    def __init__(self, model: str = None):
        from gemini_client import DEFAULT_MODEL
        self.model = model or DEFAULT_MODEL
        self.cat = render_catalogue()
        self.valid = set(labels())

    def route(self, query: str) -> str | None:
        from gemini_client import generate
        try:
            txt = generate(LIVE_PROMPT.format(catalogue=self.cat, query=query),
                           model=self.model, thinking_budget=0,
                           max_output_tokens=256, response_json=True)
            lab = json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M))["label"]
        except Exception:
            return None
        return lab if lab in self.valid else None


def _is_compiled(path: Path) -> bool:
    """A compiled rule file carries an `author`; a fitted one carries `literals`."""
    try:
        return "author" in json.loads(path.read_text())
    except Exception:
        return False


for _p in sorted(HERE.glob("rules_*.json")):
    if _is_compiled(_p):
        register(f"compiled-{_p.stem.replace('rules_', '')}",
                 lambda p=_p: CompiledRouter(p))
register("gemini-live", lambda: GeminiLiveArm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compile", "eval", "live", "prompt"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="wild")
    a = ap.parse_args()

    if a.cmd == "prompt":
        p = build_prompt()
        print(p[:1500] + f"\n...\n[{len(p)} chars, {len(labels())} targets]")
        return 0

    from gemini_client import DEFAULT_MODEL, available
    ok, why = available()
    if not ok and a.cmd != "eval":
        print(f"gateway unavailable — {why}", file=sys.stderr)
        return 2

    if a.cmd == "compile":
        compile_rules(a.model or DEFAULT_MODEL, a.tag)
        return 0

    from eval import load_split, score
    splits = {"family A (fitted)": HERE / "data" / "family_a.jsonl",
              "family B (held-out)": HERE / "data" / "family_b.jsonl",
              "wild (hand-authored)": HERE / "wild.jsonl"}
    if a.cmd == "live":
        arm = GeminiLiveArm(a.model)
        rows = load_split(splits[[k for k in splits if k.startswith(a.split)][0]])
        s = score(arm, rows)
        print(json.dumps({k: v for k, v in s.items() if k != "errors"}, indent=1))
        return 0

    for p in sorted(HERE.glob("rules_gemini*.json")):
        r = CompiledRouter(p)
        print(f"\n{p.name}: {len(r.rules)} rules  (author {r.meta.get('author')})")
        for sname, sp in splits.items():
            s = score(r, load_split(sp))
            print(f"  {sname:<22} acc {s['label_acc']:.3f}  tool {s['tool_acc']:.3f}  "
                  f"abst {s['abstain_acc']:.3f}  {s['median_latency_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
