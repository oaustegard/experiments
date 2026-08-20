#!/usr/bin/env python3
"""The `regex_only.py` arm: ordered rules a person writes by reading the schemas.

This exists to answer one question the parent experiment could not: is a *fitted*
decision list better or worse than the hand-written kind, on the same catalogue
and the same splits?

Two caveats, stated up front because they both flatter this arm:

1. These rules were written by the same author as the query templates. A real
   hand-writer meeting real traffic does not get that. Treat every number here
   as an optimistic bound on hand-writing.
2. Writing all 79 targets by hand is exactly the labour the fitter exists to
   avoid, and this file does not attempt it — it covers the tools and the
   methods a person would reach for first, and abstains elsewhere. That
   incompleteness is a result, not an omission.

`--fallback` turns on a catch-all `search_code` rule, which replicates
`monad-bsky`'s catch-all directly: it buys coverage and destroys abstention.
"""

from __future__ import annotations

import re
from pathlib import Path

from catalogue import load as load_catalogue
from cues import cues, extract

HERE = Path(__file__).resolve().parent

# (label, keyword pattern, required structural cue or None). First match wins.
RULES: list[tuple[str, str, str | None]] = [
    # --- things with a pull request in hand -----------------------------------
    ("pull_request_read::get_diff", r"\b(diff|patch|changeset)\b", "pr_ref"),
    ("pull_request_read::get_status", r"\b(status|green|red|passing|failing|checks? pass)\b", "pr_ref"),
    ("pull_request_read::get_check_runs", r"\bcheck runs?\b", "pr_ref"),
    ("pull_request_read::get_files", r"\bfiles?\b|\bpaths?\b|\btouch(es|ed)?\b", "pr_ref"),
    ("pull_request_read::get_commits", r"\bcommits?\b", "pr_ref"),
    ("pull_request_read::get_review_comments", r"\b(review comments?|inline|line[- ]level)\b", "pr_ref"),
    ("pull_request_read::get_reviews", r"\b(reviews?|approv(e|ed|al)|reviewed)\b", "pr_ref"),
    ("pull_request_read::get_comments", r"\b(comments?|conversation|thread|discussion)\b", "pr_ref"),
    ("merge_pull_request", r"\b(merge|land|squash)\b", "pr_ref"),
    ("update_pull_request_branch", r"\b(update the branch|sync|up to date|rebase)\b", "pr_ref"),
    ("update_pull_request", r"\b(retitle|out of draft|ready for review|edit)\b", "pr_ref"),
    ("request_copilot_review", r"\bcopilot\b", "pr_ref"),
    ("pull_request_review_write::submit_pending", r"\bsubmit\b", "pr_ref"),
    ("pull_request_review_write::delete_pending", r"\b(discard|scrap|delete|throw away)\b", "pr_ref"),
    ("pull_request_review_write::create", r"\b(start|create|begin|open)\b.*\breview\b", "pr_ref"),
    ("pull_request_read::get", r"", "pr_ref"),  # a bare PR reference is a PR read
    # --- review threads --------------------------------------------------------
    ("unresolve_review_thread", r"\b(unresolve|reopen|isn't settled|not done)\b", "thread_id"),
    ("resolve_review_thread", r"\b(resolve|resolved|settle|close)\b", "thread_id"),
    # --- issues ----------------------------------------------------------------
    ("issue_read::get_sub_issues", r"\b(sub[- ]issues?|nested under|child issues?)\b", None),
    ("issue_read::get_parent", r"\b(parent|umbrella|filed under|belongs to)\b", "issue_ref"),
    ("issue_read::get_labels", r"\blabels?\b", "issue_ref"),
    ("issue_read::get_comments", r"\bcomments?\b|\bsaid\b|\bdiscussion\b", "issue_ref"),
    ("add_issue_comment", r"\b(comment on|leave a (note|comment)|reply on)\b", "issue_ref"),
    ("assign_copilot_to_issue", r"\bcopilot\b", "issue_ref"),
    ("sub_issue_write", r"\b(sub[- ]issue|child of|nest)\b", "issue_ref"),
    ("issue_write::update", r"\b(close|closed|update|edit|retitle|reopen)\b", "issue_ref"),
    ("issue_write::create", r"\b(create|open|file|raise)\b.*\b(issue|bug|ticket)\b", None),
    ("issue_read::get", r"", "issue_ref"),
    # --- actions ---------------------------------------------------------------
    ("actions_run_trigger::rerun_failed_jobs", r"\bfailed jobs?\b|\bones that failed\b|\bred jobs?\b", None),
    ("actions_run_trigger::rerun_workflow_run", r"\bre-?run\b|\bretry\b|\bagain\b", "run_ref"),
    ("actions_run_trigger::cancel_workflow_run", r"\b(cancel|stop|abort)\b", "run_ref"),
    ("actions_run_trigger::delete_workflow_run_logs", r"\b(delete|purge|wipe|remove)\b.*\blogs?\b", "run_ref"),
    ("actions_get::get_workflow_run_usage", r"\b(usage|minutes|billable|cost)\b", "run_ref"),
    ("actions_get::get_workflow_run_logs_url", r"\b(logs? url|download .*logs?|log archive|raw logs?)\b", None),
    ("actions_list::list_workflow_run_artifacts", r"\bartifacts?\b", "run_ref"),
    ("actions_get::download_workflow_run_artifact", r"\b(download|grab|fetch|save)\b.*\bartifact\b", None),
    ("actions_list::list_workflow_jobs", r"\bjobs?\b", "run_ref"),
    ("get_job_logs", r"\blogs?\b", None),
    ("actions_get::get_workflow_run", r"", "run_ref"),
    ("actions_run_trigger::run_workflow", r"\b(run|trigger|kick off|start)\b", "workflow_file"),
    ("actions_list::list_workflow_runs", r"\bruns?\b|\brun history\b", "workflow_file"),
    ("actions_get::get_workflow", r"", "workflow_file"),
    ("actions_list::list_workflows", r"\bworkflows?\b|\bpipelines?\b", None),
    # --- files, commits, refs ---------------------------------------------------
    ("delete_file", r"\b(delete|remove|drop|get rid of)\b", "path"),
    ("push_files", r"\b(push|several files|multiple files|single commit|batch)\b", None),
    ("create_or_update_file", r"\b(create|update|write|commit)\b", "path"),
    ("get_file_contents", r"", "path"),
    ("get_commit", r"", "sha"),
    ("get_release_by_tag", r"\brelease\b", "semver_tag"),
    ("get_tag", r"\btags?\b", "semver_tag"),
    ("get_latest_release", r"\b(latest|newest|most recent|last)\b.*\brelease\b|\blast ship", None),
    ("list_releases", r"\breleases\b|\bversions\b", None),
    ("list_tags", r"\btags\b", None),
    ("create_branch", r"\b(create|make|cut|start)\b.*\bbranch\b", None),
    ("list_branches", r"\bbranch(es)?\b", None),
    ("list_commits", r"\bcommits?\b|\blanded\b|\bhistory\b", None),
    # --- repo-level -------------------------------------------------------------
    ("fork_repository", r"\bfork\b", None),
    ("create_repository", r"\b(create|make|new|spin up)\b.*\brepo(sitory)?\b", None),
    ("list_repository_collaborators", r"\b(collaborators?|who can push|has access)\b", None),
    ("list_pull_requests", r"\b(prs?|pull requests?)\b", None),
    ("list_issues", r"\bissues?\b", None),
    ("list_issue_types", r"\bissue types?\b", None),
    ("list_issue_fields", r"\bissue fields?\b", None),
    ("get_label", r"\blabel\b", None),
    # --- identity ----------------------------------------------------------------
    ("get_me", r"\b(who am i|my (github )?profile|logged in as|authenticated as)\b", None),
    ("get_teams", r"\b(my teams|teams i)\b", None),
    ("get_team_members", r"\bteam\b", "org_team"),
    # --- search -------------------------------------------------------------------
    ("search_code", r"\b(search|find|where|hunt)\b.*\bcode\b|\bin the (source|codebase)\b", None),
    ("search_commits", r"\b(search|find|dig)\b.*\bcommits?\b", None),
    ("search_issues", r"\b(search|find|any)\b.*\b(issues?|tickets?)\b", None),
    ("search_pull_requests", r"\b(search|find|which)\b.*\b(prs?|pull requests?)\b", None),
    ("search_repositories", r"\b(search|find|which)\b.*\brepos(itories)?\b|\bprojects out there\b", None),
    ("search_users", r"\b(users?|people|accounts?)\b", None),
]

FALLBACK = "search_code"


class HandRouter:
    def __init__(self, fallback: bool = False):
        self.catalogue = load_catalogue("session")
        self.fallback = fallback
        self.rules = [(l, re.compile(p, re.I) if p else None, c) for l, p, c in RULES]

    def route(self, q: str) -> str | None:
        c = cues(q)
        for label, rx, need in self.rules:
            if need and not c.get(need):
                continue
            if rx and not rx.search(q):
                continue
            return label
        return FALLBACK if self.fallback else None

    def call(self, q: str) -> dict | None:
        label = self.route(q)
        if label is None:
            return None
        tool, _, method = label.partition("::")
        spec = self.catalogue[tool]
        args = {k: v for k, v in extract(q).items() if k in spec["params"]}
        if method:
            args["method"] = method
        return {"tool": tool, "method": method or None, "args": args,
                "missing_required": [k for k in spec["required"] if k not in args]}


if __name__ == "__main__":
    import sys
    r = HandRouter()
    for q in sys.argv[1:] or ["show me the diff for PR 412 in cli/cli"]:
        print(f"{q}\n  -> {r.call(q)}")
