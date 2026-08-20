#!/usr/bin/env python3
"""Edit an ordered regex rule list by anchor rather than by index.

The iterated arm's failure mode, as `gh-mcp-regex-fit/RESULTS.md` diagnoses it,
is that "each revision rewrites the whole ordered list, so a fix inserted for one
error" displaces rules that were already right. Any revision procedure therefore
needs to make *where* a rule lands explicit and checkable, not implicit in a
regenerated blob.

These helpers do that. A rule is addressed by `(label, requires)` — the pair that
is unique in practice in these lists — so an edit reads as "put the PR-scoped
resolve_thread rules ahead of the bare-thread ones" instead of "insert at 49".
Every operation raises if its anchor is missing or ambiguous, so a revision that
silently no-ops is impossible.

Not specific to this experiment beyond the rule schema
(`{label, pattern, requires}`), which is `gemini_arms.CompiledRouter`'s.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

Rule = dict


def load(path: Path) -> tuple[dict, list[Rule]]:
    blob = json.loads(Path(path).read_text())
    meta = {k: v for k, v in blob.items() if k != "rules"}
    return meta, list(blob["rules"])


def find(rules: list[Rule], label: str, requires: str | None) -> int:
    hits = [i for i, r in enumerate(rules)
            if r["label"] == label and (r.get("requires") or None) == requires]
    if len(hits) != 1:
        raise KeyError(f"anchor ({label!r}, {requires!r}) matched {len(hits)} rules, need 1")
    return hits[0]


def rule(label: str, pattern: str, requires: str | None = None) -> Rule:
    re.compile(pattern, re.I)          # fail loudly here, not silently at load
    return {"label": label, "pattern": pattern, "requires": requires}


def insert_before(rules: list[Rule], anchor: tuple[str, str | None],
                  new: list[Rule]) -> list[Rule]:
    i = find(rules, *anchor)
    return rules[:i] + list(new) + rules[i:]


def insert_after(rules: list[Rule], anchor: tuple[str, str | None],
                 new: list[Rule]) -> list[Rule]:
    i = find(rules, *anchor)
    return rules[:i + 1] + list(new) + rules[i + 1:]


def repattern(rules: list[Rule], anchor: tuple[str, str | None], pattern: str) -> list[Rule]:
    i = find(rules, *anchor)
    re.compile(pattern, re.I)
    out = list(rules)
    out[i] = {**out[i], "pattern": pattern}
    return out


def move_before(rules: list[Rule], moving: list[tuple[str, str | None]],
                anchor: tuple[str, str | None]) -> list[Rule]:
    """Relocate rules without rewriting them. Order among `moving` is preserved."""
    idx = [find(rules, *m) for m in moving]
    taken = [rules[i] for i in idx]
    rest = [r for i, r in enumerate(rules) if i not in set(idx)]
    j = find(rest, *anchor)
    return rest[:j] + taken + rest[j:]


def drop(rules: list[Rule], anchor: tuple[str, str | None]) -> list[Rule]:
    i = find(rules, *anchor)
    return rules[:i] + rules[i + 1:]


def save(path: Path, meta: dict, rules: list[Rule], **extra) -> Path:
    for r in rules:
        re.compile(r["pattern"] or "", re.I)
    blob = {**meta, "n_returned": len(rules), "n_kept": len(rules),
            "invented_labels": [], **extra, "rules": rules}
    Path(path).write_text(json.dumps(blob, indent=1) + "\n")
    return Path(path)
