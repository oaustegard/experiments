#!/usr/bin/env python3
"""Structural cues and argument extractors for the GitHub MCP catalogue.

Two jobs, deliberately separated:

* `cues(query)` returns the boolean structural facts a router may condition on
  — "there is a pull-request reference in here", "there is a 40-hex sha". These
  are the features the fitter is allowed to build rules from. They come from the
  *shape* of the string, never from a keyword.
* `extract(query)` returns the spans themselves, so arguments are bound by
  copying rather than by generation. This is `monad-bsky/repair.py` generalised:
  a value that appears verbatim in the request should never be retyped.

Every cue is derived from a parameter that actually exists in the catalogue
(`pullNumber`, `sha`, `path`, `run_id`, `tag`, `threadID`, ...), so the cue set
is a function of the schemas, not of any query set.

    python3 cues.py "what changed in oaustegard/experiments#412"
"""

from __future__ import annotations

import re
import sys

# --- URL forms. Ordered longest-first; each also implies a plainer cue. -------
RX = {
    "url_pull": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", re.I),
    "url_issue": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/issues/(\d+)", re.I),
    "url_blob": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/blob/([\w.\-/]+?)/(\S+)", re.I),
    "url_commit": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/commit/([0-9a-f]{7,40})", re.I),
    "url_run": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/actions/runs/(\d+)", re.I),
    "url_release": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/releases/tag/(\S+)", re.I),
    "url_repo": re.compile(r"github\.com/([\w.-]+)/([\w.-]+)/?(?:\s|$)", re.I),
}

# --- bare tokens -------------------------------------------------------------
SHA40 = re.compile(r"\b[0-9a-f]{40}\b", re.I)
# A short sha must contain a hex letter: an all-digit run of 7+ is a run id.
SHA_SHORT = re.compile(r"\b(?=[0-9a-f]{7,12}\b)[0-9]*[a-f][0-9a-f]*\b(?![\w.-])")
PR_WORD = re.compile(r"\b(?:pr|pull request|pull-request)\s*#?\s*(\d+)\b", re.I)
ISSUE_WORD = re.compile(r"\bissue\s*#?\s*(\d+)\b", re.I)
RUN_WORD = re.compile(r"\b(?:run|workflow run|job)\s*#?\s*(\d{6,})\b", re.I)
HASH_NUM = re.compile(r"(?<!&)#(\d+)\b")
OWNER_REPO = re.compile(r"(?<![\w/.])([A-Za-z][\w.-]*)/([A-Za-z][\w.-]*)(?![\w/.])")
SEMVER = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b")
TAGGY = re.compile(r"\bv\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?\b")
PATHY = re.compile(r"(?<![\w/])([\w.-]+/)*[\w.-]+\.(?:py|go|ts|tsx|js|jsx|md|json|ya?ml|toml|rs|java|rb|sh|txt|cfg|ini|lock)\b")
WORKFLOW_FILE = re.compile(r"\b[\w.-]+\.ya?ml\b")
# Branches are named on both sides of the word: "branch release/v2" and
# "the release/v2 branch" both occur, and the first form must not swallow a
# following preposition.
BRANCH_STOP = r"(?!(?:of|in|on|for|at|from|to|the|and|is|was)\b)"
BRANCHY = re.compile(
    r"\b(?:branch|onto|into|rebase(?:d)? on)\s+" + BRANCH_STOP + r"([\w.\-/]+)\b"
    r"|\b([\w.\-/]+)\s+branch\b", re.I)
AT_USER = re.compile(r"(?<![\w/])@([A-Za-z][\w-]{0,38})\b")
QUOTED = re.compile(r"[\"'“‘]([^\"'”’]{2,80})[\"'”’]")
THREAD_ID = re.compile(r"\b(?:PRRT|RT)_[\w]{8,}\b")
ORG_TEAM = re.compile(r"\b([\w-]+)\s+(?:org(?:anization)?|team)\b", re.I)
BARE_NUM = re.compile(r"(?<![\w#/.])(\d{1,6})(?![\w/.])")

# Words that look like `owner/repo` but are prose. Kept tiny on purpose: this is
# the one place a lexical list leaks into the structural layer, and every entry
# is a generic English construction rather than anything from the query set.
NOT_A_REPO = {
    "and/or", "he/she", "i/o", "n/a", "read/write", "yes/no", "true/false",
    "ci/cd", "a/b", "w/o", "front/back", "input/output", "pass/fail",
}


def _branch_spans(q: str) -> list[tuple[int, int]]:
    spans = []
    for m in BRANCHY.finditer(q):
        for g in (1, 2):
            if m.group(g):
                spans.append(m.span(g))
    return spans


def _owner_repos(q: str) -> list[tuple[str, str]]:
    out = []
    bspans = _branch_spans(q)
    for m in OWNER_REPO.finditer(q):
        if any(bs <= m.start() and m.end() <= be for bs, be in bspans):
            continue  # `release/v2` in "the release/v2 branch" is a ref, not a repo
        whole = m.group(0)
        if whole.lower() in NOT_A_REPO:
            continue
        if PATHY.fullmatch(whole) or "." in m.group(2) and m.group(2).split(".")[-1].isalpha() and len(m.group(2).split(".")[-1]) <= 4 and m.group(2).count(".") >= 1:
            # `src/router.py` is a path, not a repo
            continue
        out.append((m.group(1), m.group(2)))
    return out


def extract(q: str) -> dict:
    """Every structurally recoverable value in the query, bound by name."""
    v: dict[str, object] = {}

    for kind, rx in RX.items():
        m = rx.search(q)
        if not m:
            continue
        v.setdefault("owner", m.group(1))
        v.setdefault("repo", m.group(2))
        if kind == "url_pull":
            v.setdefault("pullNumber", int(m.group(3)))
        elif kind == "url_issue":
            v.setdefault("issue_number", int(m.group(3)))
        elif kind == "url_blob":
            v.setdefault("ref", m.group(3))
            v.setdefault("path", m.group(4))
        elif kind == "url_commit":
            v.setdefault("sha", m.group(3))
        elif kind == "url_run":
            v.setdefault("run_id", int(m.group(3)))
        elif kind == "url_release":
            v.setdefault("tag", m.group(3))

    if "owner" not in v:
        ors = _owner_repos(q)
        if ors:
            v["owner"], v["repo"] = ors[0]

    if m := PR_WORD.search(q):
        v.setdefault("pullNumber", int(m.group(1)))
    if m := ISSUE_WORD.search(q):
        v.setdefault("issue_number", int(m.group(1)))
    if m := RUN_WORD.search(q):
        v.setdefault("run_id", int(m.group(1)))
    if m := SHA40.search(q):
        v.setdefault("sha", m.group(0))
    elif m := SHA_SHORT.search(q):
        v.setdefault("sha", m.group(0))
    if m := TAGGY.search(q):
        v.setdefault("tag", m.group(0))
    if m := PATHY.search(q):
        v.setdefault("path", m.group(0))
    if m := THREAD_ID.search(q):
        v.setdefault("threadID", m.group(0))
    if m := AT_USER.search(q):
        v.setdefault("login", m.group(1))
    if m := QUOTED.search(q):
        v.setdefault("query", m.group(1))
    if m := BRANCHY.search(q):
        v.setdefault("branch", m.group(1) or m.group(2))

    # `#412` is ambiguous between a PR and an issue: bind it to both slots and
    # let the routing decision pick which one the tool wants.
    if m := HASH_NUM.search(q):
        n = int(m.group(1))
        v.setdefault("pullNumber", n)
        v.setdefault("issue_number", n)
    return v


CUE_NAMES = [
    "url_pull", "url_issue", "url_blob", "url_commit", "url_run", "url_release",
    "url_repo", "pr_ref", "issue_ref", "hash_num", "sha", "sha40", "run_ref",
    "path", "workflow_file", "semver_tag", "owner_repo", "at_user", "quoted",
    "thread_id", "org_team", "bare_num", "branch_word",
]


def cues(q: str) -> dict[str, bool]:
    """Boolean structural facts. The only non-lexical features the fitter sees."""
    c = {k: bool(rx.search(q)) for k, rx in RX.items()}
    c["pr_ref"] = bool(PR_WORD.search(q)) or c["url_pull"]
    c["issue_ref"] = bool(ISSUE_WORD.search(q)) or c["url_issue"]
    c["hash_num"] = bool(HASH_NUM.search(q))
    c["sha40"] = bool(SHA40.search(q))
    c["sha"] = c["sha40"] or bool(SHA_SHORT.search(q)) or c["url_commit"]
    c["run_ref"] = bool(RUN_WORD.search(q)) or c["url_run"]
    c["path"] = bool(PATHY.search(q)) or c["url_blob"]
    c["workflow_file"] = bool(WORKFLOW_FILE.search(q))
    c["semver_tag"] = bool(TAGGY.search(q)) or c["url_release"]
    c["owner_repo"] = bool(_owner_repos(q)) or c["url_repo"] or any(
        c[k] for k in ("url_pull", "url_issue", "url_blob", "url_commit", "url_run")
    )
    c["at_user"] = bool(AT_USER.search(q))
    c["quoted"] = bool(QUOTED.search(q))
    c["thread_id"] = bool(THREAD_ID.search(q))
    c["org_team"] = bool(ORG_TEAM.search(q))
    c["bare_num"] = bool(BARE_NUM.search(q))
    c["branch_word"] = bool(BRANCHY.search(q))
    return {k: c.get(k, False) for k in CUE_NAMES}


def main() -> int:
    q = " ".join(sys.argv[1:]) or "what changed in oaustegard/experiments#412"
    print(q)
    print("  cues:   ", ", ".join(k for k, v in cues(q).items() if v) or "(none)")
    print("  extract:", extract(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
