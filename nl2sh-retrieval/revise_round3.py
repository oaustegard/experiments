#!/usr/bin/env python3
"""Round 3 of the Claude iterated arm — and the round the protocol cannot supply.

Scoring `rules_claude-iter2.json` on family A gives **1.000, zero errors**, so
the round-3 prompt this arm mirrors would carry an empty error list. That is not
the situation the Gemini arm was in: its round-2 rules scored 0.912, so round 3
still had ~88 errors to sample from, and it lost 0.123 in sample by rewriting
around them. Claude never enters that regime, so "did Claude degrade at round 3"
has to be answered about two different objects:

* **`--faithful` (default)** — what the reviser actually does when shown no
  errors: nothing. The output is byte-identical in rule content to round 2. This
  is the honest round-3 artifact and the one in the four-round table.

* **`--speculative`** — what a reviser does if it edits *anyway*, which is what a
  model handed "keep what works, fix what does not" and an empty error list will
  in practice do. Ten broadening edits, each a synonym or paraphrase a person
  would endorse from the language alone, none of them prompted by any observed
  failure. This is the direct test of whether continued revision without signal
  costs held-out accuracy, which is the mechanism the Gemini regression was
  attributed to. It is reported as a diagnostic arm, not as round 3.

Neither variant looks at family B or the wild set.

    python3 revise_round3.py                 # faithful
    python3 revise_round3.py --speculative
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ruleops import load, repattern, save

HERE = Path(__file__).resolve().parent


def revise_speculative(rules: list[dict]) -> list[dict]:
    """Broaden ten rules on language intuition alone. No error motivated any of these."""
    edits = [
        # more ways to ask for a diff
        (("pull_request_read::get_diff", "pr_ref"),
         r"\b(diff|patch|changeset|change ?set|delta)\b|\bwhat changed\b|\bcode changes?\b|"
         r"\bwhat('s| is| are)? ?(the )?(code )?changes?\b|\bwhat does (it|this|the pr) change\b|"
         r"\bshow (me )?(the )?changes?\b|\bwhat did .{0,25}change\b|\bunified diff\b|"
         r"\bwhat code\b.{0,25}\bchange\w*\b|\bwhat('s| is) in (it|this|the pr)\b|"
         r"\bhow does (it|this) differ\b|\bline by line\b"),
        # more ways to ask which files moved
        (("pull_request_read::get_files", "pr_ref"),
         r"\b(which|what|list|show|all|the|how many|any)\b.{0,30}\bfiles?\b|"
         r"\bfiles?\b.{0,30}\b(changed|touched|modified|affected|added|removed|renamed|involved)\b|"
         r"\b(changed|touched|modified|affected)\b.{0,20}\bfiles?\b|\bfile list\b|"
         r"\bblast radius\b|\bsurface area\b|\bwhat did it touch\b"),
        # more ways to say "land it"
        (("merge_pull_request", "pr_ref"),
         r"\b(merge|squash|rebase and merge|land|ship it|merge it|auto-?merge)\b|"
         r"\bget (it|this) (in|merged|landed)\b|\bpull the trigger\b|\bsend it\b"),
        # more ways to ask if CI is happy
        (("pull_request_read::get_status", "pr_ref"),
         r"\bstatus(es)?\b|\bgreen\b|\bpassing\b|\bfailing\b|\bmergeable\b|\bbuild state\b|"
         r"\bci\b|\bhealthy\b|\ball clear\b|\bsafe to merge\b|"
         r"\b(is|are) (it|this|that|they) (ok|good|clean|red)\b"),
        # more ways to name the backlog
        (("list_issues", "owner_repo"),
         r"\b(list|show|what|which|any|all|open|closed|see|get|are there)\b.{0,40}\b(issues|tickets|bugs)\b|"
         r"\b(open|closed) issues\b|\bbacklog\b|\bwhat needs doing\b"),
        # more ways to ask for code search
        (("search_code", None),
         r"\b(search|find|grep|look for|locate|where('s| is)|which files?|any)\b.{0,60}"
         r"\b(code|function|func|method|class|symbol|usages?|uses|definition|defined|implementation|"
         r"implements?|string|variable|const|import|calls?|called|snippet|pattern|regex|todo)\b|"
         r"\bgrep\b|\bcode search\b|\bsearch (the )?(code|codebase|source)\b|"
         r"\bwhere (is|are)\b.{0,30}\b(defined|implemented|used)\b|\bfind where\b|\bwho calls\b"),
        # more ways to ask for logs
        (("get_job_logs", "run_ref"),
         r"\blogs?\b|\blog output\b|\bstack ?trace\b|\btraceback\b|\berror (output|message|log)\b|"
         r"\bwhy\b.{0,40}\bfail\w*\b|\bwhat went wrong\b|\bfail\w*\b.{0,30}\b(output|reason|why|message)\b|"
         r"\boutput\b.{0,25}\bfail\w*\b|\bconsole\b|\bstderr\b|\bwhat did it print\b"),
        # more ways to open a PR
        (("create_pull_request", None),
         r"\b(open|create|make|raise|file|submit|start|put up|send|draft)\b.{0,40}\b(pull request|prs?|p\.r\.)\b|"
         r"\bpr\b.{0,25}\bfrom\b.{0,30}\b(into|to|against)\b|\bmake (this|it) a pr\b|"
         r"\bput up a pr\b|\bpropose (these|the) changes\b"),
        # more ways to name a branch listing
        (("list_branches", None),
         r"\b(list|show|what|which|all|any|see)\b.{0,40}\bbranches\b|"
         r"\bbranches\b.{0,30}\b(exist|are there|does|do)\b|\bbranch list\b|\bwhat branches\b"),
        # more ways to ask who reviewed
        (("pull_request_read::get_reviews", "pr_ref"),
         r"\breviews?\b|\bwho (approved|reviewed|has reviewed|signed off)\b|\bapprovals?\b|"
         r"\bapproved\b|\breviewers?\b|\bhas (it|this) been reviewed\b|\bsign-?offs?\b|"
         r"\bgot (a |any )?(thumbs|approval)\b"),
    ]
    for anchor, pat in edits:
        rules = repattern(rules, anchor, pat)
    return rules


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("../gh-mcp-regex-fit/rules_claude-iter2.json"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--speculative", action="store_true")
    a = ap.parse_args()
    src = a.src if a.src.is_absolute() else HERE / a.src
    default = ("../gh-mcp-regex-fit/rules_claude-iter3-speculative.json" if a.speculative
               else "../gh-mcp-regex-fit/rules_claude-iter3.json")
    out = a.out or Path(default)
    out = out if out.is_absolute() else HERE / out
    meta, rules = load(src)
    if a.speculative:
        new = revise_speculative(rules)
        note = ("round 3 diagnostic; family A had ZERO errors, so these 10 broadening "
                "edits are motivated by language intuition alone, not by any failure")
    else:
        new = rules
        note = ("round 3; 0 family-A errors shown (family A saturated at 1.000 after "
                "round 2), so the revision is the identity")
    save(out, meta, new, supervision=note, parent=src.name)
    print(f"{len(rules)} -> {len(new)} rules, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
