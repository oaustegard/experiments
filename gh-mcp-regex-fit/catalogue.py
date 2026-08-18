#!/usr/bin/env python3
"""Build the GitHub MCP tool catalogue from the upstream server's own schema snapshots.

`github/github-mcp-server` commits a JSON snapshot of every tool's MCP schema
under `pkg/github/__toolsnaps__/`. Those are the exact bytes an MCP client is
handed, so the catalogue here is the real one rather than a reconstruction.

    python3 catalogue.py --clone     # fetch upstream, rebuild catalogue.json
    python3 catalogue.py             # summarise the committed catalogue.json

`SESSION_TOOLS` is the 58-tool subset actually exposed to a Claude Code on the
Web session on 2026-08-18 (read off the harness tool list). The full snapshot
set is larger because it includes toolsets this deployment does not enable and
the granular variants that `issue_read` / `pull_request_read` replaced.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOGUE = HERE / "catalogue.json"
UPSTREAM = "https://github.com/github/github-mcp-server.git"

# The 58 mcp__github__* tools exposed to this session, verbatim.
SESSION_TOOLS = """
actions_get actions_list actions_run_trigger add_comment_to_pending_review
add_issue_comment add_reply_to_pull_request_comment assign_copilot_to_issue
create_branch create_or_update_file create_pull_request
create_pull_request_with_copilot create_repository delete_file
disable_pr_auto_merge enable_pr_auto_merge fork_repository get_check_run
get_commit get_copilot_job_status get_file_contents get_job_logs get_label
get_latest_release get_me get_release_by_tag get_tag get_team_members get_teams
issue_read issue_write list_branches list_commits list_issue_fields
list_issue_types list_issues list_pull_requests list_releases
list_repository_collaborators list_tags merge_pull_request pull_request_read
pull_request_review_write push_files request_copilot_review
resolve_review_thread run_secret_scanning search_code search_commits
search_issues search_pull_requests search_repositories search_users
sub_issue_write subscribe_pr_activity unresolve_review_thread
unsubscribe_pr_activity update_pull_request update_pull_request_branch
""".split()


def _clean(text: str) -> str:
    """Collapse whitespace; drop the long prose blocks some descriptions carry."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def build(snap_dir: Path) -> dict:
    tools = {}
    for path in sorted(snap_dir.glob("*.snap")):
        snap = json.loads(path.read_text())
        name = snap["name"]
        schema = snap.get("inputSchema", {})
        props = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []
        tools[name] = {
            "name": name,
            "title": _clean(snap.get("annotations", {}).get("title", "")),
            "read_only": bool(snap.get("annotations", {}).get("readOnlyHint", False)),
            "description": _clean(snap.get("description", ""))[:600],
            "required": list(required),
            "params": {
                pname: {
                    "type": p.get("type", "string"),
                    "required": pname in required,
                    "enum": p.get("enum"),
                    "description": _clean(p.get("description", ""))[:240],
                }
                for pname, p in sorted(props.items())
            },
        }
    return tools


def load(subset: str = "session") -> dict:
    tools = json.loads(CATALOGUE.read_text())["tools"]
    if subset == "all":
        return tools
    return {n: t for n, t in tools.items() if n in set(SESSION_TOOLS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true", help="re-fetch upstream snapshots")
    ap.add_argument("--src", type=Path, help="existing checkout of github-mcp-server")
    a = ap.parse_args()

    if a.clone or a.src:
        with tempfile.TemporaryDirectory() as tmp:
            root = a.src or Path(tmp) / "gh-mcp"
            if not a.src:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "-q", UPSTREAM, str(root)], check=True
                )
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            tools = build(root / "pkg" / "github" / "__toolsnaps__")
        CATALOGUE.write_text(
            json.dumps({"source": UPSTREAM, "commit": head, "tools": tools}, indent=1) + "\n"
        )
        print(f"wrote {CATALOGUE.name}: {len(tools)} tools @ {head[:12]}")

    tools = json.loads(CATALOGUE.read_text())["tools"]
    session = load("session")
    missing = sorted(set(SESSION_TOOLS) - set(tools))
    print(f"snapshots: {len(tools)}   session subset: {len(session)}/{len(SESSION_TOOLS)}")
    if missing:
        print(f"  not in upstream snapshots ({len(missing)}): {', '.join(missing)}")
    ro = sum(t["read_only"] for t in session.values())
    print(f"  read-only {ro}, mutating {len(session) - ro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
