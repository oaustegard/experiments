#!/usr/bin/env python3
"""Round 1 of the Claude iterated arm: revise the clean-room rules from 120 family-A errors.

Every edit below is traceable to a cluster in `errors_round1.json` and to a
mechanism, not to a memorised query. The clusters, largest first:

 1. `job 785745729` is a `run_ref` (cues.RUN_WORD accepts `job`), so the broad
    `get_workflow_run` rule swallows it before the job rule is reached.
 2. `resolve review thread PRRT_x on PR 20` — the bare `resolve_review_thread`
    tool takes only a threadID; the `pull_request_review_write::resolve_thread`
    method also takes a pull number. When the request names a PR the method is
    the right target, so the PR-scoped rules must precede the bare ones.
 3. `\b(re-?run|retry)\b.{0,10}$` in the rerun rule fires on any request ending
    near the word "retry" — `search users for retry backoff`, `on branch
    feature/retry`, `fix the retry loop`. One runaway alternation, ~20 errors
    spread over nine unrelated targets.
 4. `#1423` sets `hash_num` but neither `pr_ref` nor `issue_ref`, and *every*
    pull-request rule is conditioned on `pr_ref`/`url_pull`. So the whole PR
    family is unreachable for bare-hash requests and they fall through to the
    issue rules or abstain. Family A's hash-only rows are ~80% pull-request
    targets, and the disambiguator is the vocabulary of the request (diff,
    reviews, checks, merge) rather than the reference itself.
 5. `\bunder issue\b` in `sub_issue_write` fires on *reads* — "get the sub
    issues under issue 2498" — because it tests the preposition and not the
    verb.
 6. `issue_read::get_parent` tests only the literal word "parent" and misses
    "which issue X belongs to".
 7. `push_files` sits after `create_or_update_file`, so a multi-file push that
    names one path is routed as a single-file write.
 8. `issue_write::update` misses "to closed" — the pattern has `\bclose\b`,
    which does not match "closed".

    python3 revise_round1.py --out ../gh-mcp-regex-fit/rules_claude-iter1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ruleops import drop, insert_after, insert_before, load, move_before, repattern, rule, save

HERE = Path(__file__).resolve().parent

# Reused verbatim from the clean-room rules so the hash-scoped copies cannot
# drift from the pr_ref-scoped originals they mirror.
PAT_DIFF = (r"\b(diff|patch|changeset|change ?set|delta)\b|\bwhat changed\b|\bcode changes?\b|"
            r"\bwhat('s| is| are)? ?(the )?(code )?changes?\b|\bwhat does (it|this|the pr) change\b|"
            r"\bshow (me )?(the )?changes?\b|\bwhat did .{0,25}change\b|\bunified diff\b|"
            r"\bwhat code\b.{0,25}\bchange\w*\b")
PAT_FILES = (r"\b(which|what|list|show|all|the|how many|any)\b.{0,30}\bfiles?\b|"
             r"\bfiles?\b.{0,30}\b(changed|touched|modified|affected|added|removed|renamed|involved)\b|"
             r"\b(changed|touched|modified|affected)\b.{0,20}\bfiles?\b|\bfile list\b")
PAT_REVCOM = (r"\b(review|inline|line|code)[ -]comments?\b|"
              r"\bcomments?\b.{0,30}\b(on the (diff|code|lines?)|in the (diff|review|code))\b|"
              r"\breviewers?'? comments?\b|\bnits?\b")
PAT_REVIEWS = (r"\breviews?\b|\bwho (approved|reviewed|has reviewed|signed off)\b|\bapprovals?\b|"
               r"\bapproved\b|\breviewers?\b|\bhas (it|this) been reviewed\b")
PAT_STATUS = (r"\bstatus(es)?\b|\bgreen\b|\bpassing\b|\bfailing\b|\bmergeable\b|\bbuild state\b|"
              r"\bci\b.{0,25}\b(pass\w*|fail\w*|green|ok|red|status|happy)\b|"
              r"\b(is|are) (it|this|that|they) (ok|good|clean|red)\b")
PAT_REVIEW_CREATE = (r"\b(approve|approving|approval|lgtm|sign ?off|request(ing)? changes|reject|block)\b|"
                     r"\b(leave|add|create|start|do|write|give|submit|post)\b.{0,25}\b(a |an |my )?review\b|"
                     r"\breview (it|this|that|the pr)\b")
PAT_UPD_BRANCH = (r"\b(update|sync|refresh|rebase|bring)\b.{0,40}\b(branch|with (main|master|base|develop|trunk)|up ?to ?date)\b|"
                  r"\bup to date with\b|\bsync\b.{0,30}\bbase\b|\bmerge (main|master|the base)\b.{0,25}\binto\b")
PAT_UPD_PR = (r"\b(close|closes|reopen|re-open)\b|"
              r"\b(rename|retitle|edit|update|change|set|convert|mark|make|move|switch|point)\b.{0,50}"
              r"\b(title|body|description|base|draft|ready for review|state|milestone|reviewers?|assignees?|labels?|target|branch)\b|"
              r"\bretarget\b|\bconvert\b.{0,25}\bdraft\b")
PAT_THREAD_UNRES = r"\bun-?resolve\w*\b|\breopen\b.{0,30}\b(thread|conversation|discussion)\b"
PAT_THREAD_RES = (r"\bresolve\w*\b.{0,40}\b(thread|conversation|comment|discussion|feedback|nit)\w*\b|"
                  r"\bmark\b.{0,40}\bresolved\b|\bresolve\b.{0,15}\b(all|the|that|this|those|every|them)\b")


def revise(rules: list[dict]) -> list[dict]:
    # (3) The runaway tail. Keep the intent — a bare "rerun it" with no object —
    #     but require the whole request to be that, rather than the last ten
    #     characters of any sentence.
    rules = repattern(
        rules, ("actions_run_trigger::rerun_workflow_run", None),
        r"\b(re-?run|re-?try|retry|re-?trigger|re-?start|re-?launch)\b.{0,40}"
        r"\b(workflow|run|build|pipeline|ci|job|action|check|deploy\w*|it|that|this)\b|"
        r"^\s*(re-?run|re-?try|retry)\s*(it|that|this)?\s*[.!?]?\s*$")

    # (1) A job id is a run_ref, so the job rule has to precede the run rule.
    rules = insert_before(rules, ("actions_get::get_workflow_run", "run_ref"), [
        rule("actions_get::get_workflow_job",
             r"\bjob\b\s*#?\s*\d{3,}\b|\bworkflow job\b|\bjob id\b", "run_ref"),
    ])

    # (2) A named PR promotes the bare thread tools to the dispatcher methods.
    rules = move_before(rules, [
        ("pull_request_review_write::unresolve_thread", "pr_ref"),
        ("pull_request_review_write::unresolve_thread", "url_pull"),
        ("pull_request_review_write::resolve_thread", "pr_ref"),
        ("pull_request_review_write::resolve_thread", "url_pull"),
    ], ("unresolve_review_thread", "thread_id"))
    rules = insert_before(rules, ("unresolve_review_thread", "thread_id"), [
        rule("pull_request_review_write::unresolve_thread", PAT_THREAD_UNRES, "hash_num"),
        rule("pull_request_review_write::resolve_thread", PAT_THREAD_RES, "hash_num"),
    ])

    # (4a) "request a copilot review on #2378" — copilot + review is the PR tool.
    rules = insert_before(rules, ("assign_copilot_to_issue", "hash_num"), [
        rule("request_copilot_review",
             r"\bcopilot\b.{0,25}\breview\b|\breview\b.{0,25}\bcopilot\b", "hash_num"),
    ])

    # (7) A multi-file push outranks the single-file write even when a path shows.
    rules = insert_before(rules, ("delete_file", "path"), [
        rule("push_files",
             r"\bpush\b.{0,40}\b(multiple|several|all|these|those|both)\b.{0,25}\bfiles?\b|"
             r"\bin (one|a single) commit\b|\bcommit\b.{0,30}\b(these|those|multiple|several|both)\b.{0,25}\bfiles?\b"),
    ])

    # (5) sub_issue_write is a write: gate it on the verb, not on "under issue".
    rules = repattern(rules, ("sub_issue_write", "issue_ref"),
                      r"\b(add|link|attach|make|set|create|nest|convert|move|file|put)\b.{0,50}"
                      r"\b(sub-? ?issue|under|child of|parent)\b|"
                      r"\bas a sub-? ?issue\b|\bsub-? ?issue of\b")

    # (6) "which issue X belongs to" is the parent query without the word parent.
    for req in ("issue_ref", "url_issue"):
        rules = repattern(rules, ("issue_read::get_parent", req),
                          r"\bparent\b|\bbelongs to\b|\bwhich issue\b.{0,40}\b(belongs|is .{0,15}under)\b")

    # (8) "\bclose\b" does not match "closed".
    for req in ("issue_ref", "url_issue"):
        rules = repattern(
            rules, ("issue_write::update", req),
            r"\b(close|closed|closing|closes|reopen|re-?open(ed)?)\b|"
            r"\b(rename|retitle|edit|update|change|set|mark|move|convert|assign|unassign|add|remove|apply|attach|put|give|drop)\b.{0,50}"
            r"\b(title|body|description|labels?|assignees?|milestone|state|type|priority|status)\b|"
            r"\bassign\b.{0,30}\bto\b|\bmark\b.{0,25}\b(done|complete|completed|duplicate|wontfix|stale)\b")
    # The hash-scoped copy has to insist on the word "issue", because a bare
    # `#N` with an edit verb is more often a PR (see 4).
    rules = repattern(
        rules, ("issue_write::update", "hash_num"),
        r"\bissue\b.{0,60}\b(close|closed|reopen|title|body|description|label|assignee|milestone|state|type|priority|status)\w*\b|"
        r"\b(edit|update|change|close|closed|reopen)\b.{0,25}\bissue\b")

    # (4b) The pull-request family, reachable on a bare `#N`. Ordered so the
    #      specific reads precede the generic PR edit, and placed after the
    #      issue-comment and issue-edit rules so genuinely issue-shaped requests
    #      still win.
    rules = insert_before(rules, ("issue_read::get", "hash_num"), [
        rule("issue_read::get_sub_issues",
             r"\bsub-? ?issues\b|\bchild(ren)? issues?\b|\bsub-?tasks?\b|\bbreakdown\b", "hash_num"),
        rule("issue_read::get_parent",
             r"\bparent\b|\bbelongs to\b", "hash_num"),
        rule("update_pull_request_branch", PAT_UPD_BRANCH, "hash_num"),
        rule("merge_pull_request",
             r"\b(merge|squash|rebase and merge|land|ship it|merge it|auto-?merge)\b", "hash_num"),
        rule("pull_request_read::get_diff", PAT_DIFF, "hash_num"),
        rule("pull_request_read::get_files", PAT_FILES, "hash_num"),
        rule("pull_request_read::get_commits", r"\bcommits\b|\bcommit list\b|\bhow many commits\b", "hash_num"),
        rule("pull_request_read::get_review_comments", PAT_REVCOM, "hash_num"),
        rule("pull_request_read::get_check_runs", r"\bcheck ?runs?\b|\bchecks\b", "hash_num"),
        rule("pull_request_read::get_status", PAT_STATUS, "hash_num"),
        rule("pull_request_read::get_reviews", PAT_REVIEWS, "hash_num"),
        rule("pull_request_review_write::create", PAT_REVIEW_CREATE, "hash_num"),
        rule("pull_request_read::get_comments",
             r"\bcomments?\b|\bdiscussion\b|\bconversation\b|\bfeedback\b|\bthreads?\b", "hash_num"),
        rule("update_pull_request", PAT_UPD_PR, "hash_num"),
    ])

    # "merge the base branch into PR 1173" is an update-branch, not a merge:
    # the branch rule must precede the merge rule for the pr_ref/url_pull forms
    # too, exactly as it now does for hash_num.
    rules = move_before(rules, [
        ("update_pull_request_branch", "pr_ref"),
        ("update_pull_request_branch", "url_pull"),
    ], ("merge_pull_request", "pr_ref"))
    return rules


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("../gh-mcp-regex-fit/rules_claude-cleanroom.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("../gh-mcp-regex-fit/rules_claude-iter1.json"))
    a = ap.parse_args()
    src = a.src if a.src.is_absolute() else HERE / a.src
    out = a.out if a.out.is_absolute() else HERE / a.out
    meta, rules = load(src)
    new = revise(rules)
    save(out, {**meta, "author": "claude-opus-5 (session, iterated reviser)"}, new,
         supervision="round 1; 120 family-A errors shown; family A only",
         parent=src.name)
    print(f"{len(rules)} -> {len(new)} rules, wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
