#!/usr/bin/env python3
"""Round 2 of the Claude iterated arm: five remaining family-A errors, two causes.

Round 1 took family A from 0.863 to 0.995, so this round has almost no signal —
which is precisely the regime where the Gemini arm began to regress. Both
surviving clusters are local:

 1. `link <issue-url> under parent issue 408440870` — the `sub_issue_write`
    pattern allows 50 characters between the verb and "under", and a full issue
    URL is 55. A gap width, not a missing alternative.
 2. `create a pending review on #2078` — the hash-scoped block I added in round
    1 put `get_reviews` ahead of `pull_request_review_write::create`, so the
    bare word "review" claimed the request before the create verb was tested.
    The pr_ref-scoped copies already have the opposite order; this restores it.

Nothing else is touched. The temptation at this point is to keep editing rules
that are not in the error list, which is what "revise from your errors" stops
being once the errors run out.

    python3 revise_round2.py --out ../gh-mcp-regex-fit/rules_claude-iter2.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ruleops import insert_before, load, move_before, repattern, rule, save

HERE = Path(__file__).resolve().parent


def revise(rules: list[dict]) -> list[dict]:
    # (1) widen the verb->preposition gap past the length of an issue URL, and
    #     give the rule the url_issue form it never had.
    wide = (r"\b(add|link|attach|make|set|create|nest|convert|move|file|put)\b.{0,80}"
            r"\b(sub-? ?issue|under|child of|parent)\b|"
            r"\bas a sub-? ?issue\b|\bsub-? ?issue of\b")
    rules = repattern(rules, ("sub_issue_write", "issue_ref"), wide)
    rules = insert_before(rules, ("issue_read::get_sub_issues", "issue_ref"),
                          [rule("sub_issue_write", wide, "url_issue")])

    # (2) creating a review outranks reading reviews, on hash_num as elsewhere.
    rules = move_before(rules, [("pull_request_review_write::create", "hash_num")],
                        ("pull_request_read::get_review_comments", "hash_num"))
    return rules


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("../gh-mcp-regex-fit/rules_claude-iter1.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("../gh-mcp-regex-fit/rules_claude-iter2.json"))
    a = ap.parse_args()
    src = a.src if a.src.is_absolute() else HERE / a.src
    out = a.out if a.out.is_absolute() else HERE / a.out
    meta, rules = load(src)
    new = revise(rules)
    save(out, meta, new,
         supervision="round 2; 5 family-A errors shown; family A only",
         parent=src.name)
    print(f"{len(rules)} -> {len(new)} rules, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
