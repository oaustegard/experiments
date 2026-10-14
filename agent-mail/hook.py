#!/usr/bin/env python3
"""PostToolUse hook: push unread mail into a subagent's context after each tool call.

A subagent becomes addressable the first time it runs `mail.py join|send|recv NAME`:
the hook binds the payload's agent_id to NAME. From then on every tool call it
makes returns with its unread mail (and new peers' cards) as additionalContext,
so nobody has to poll. No board, no agent_id, or no binding: the hook prints
nothing. Any error fails open with one line on stderr.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mail  # noqa: E402

# `<tree>/mail join|send|recv NAME` (the per-board wrapper) or `mail.py ... NAME` (ACTIVE board)
CMD_RE = re.compile(r"(\S*?)(?:/mail|mail\.py)\s+(?:join|send|recv)\s+['\"]?([A-Za-z0-9_-]+)")
BINDS = Path.home() / ".claude" / "agent-mail" / "binds"


def run(payload: dict) -> dict | None:
    agent = payload.get("agent_id")
    if not agent:
        return None
    bind = BINDS / f"{agent}.json"
    cmd = (payload.get("tool_input") or {}).get("command", "") if payload.get("tool_name") == "Bash" else ""
    m = CMD_RE.search(cmd)
    if m and not bind.exists():
        tree = m.group(1)
        if tree in ("", "."):
            tree = payload.get("cwd", "")
        b = Path(tree) / ".board" if tree and (Path(tree) / ".board").is_dir() else mail.board()
        if b is None:
            return None
        bind.parent.mkdir(parents=True, exist_ok=True)
        bind.write_text(json.dumps({"board": str(b), "name": m.group(2)}))
    if not bind.exists():
        return None
    rec = json.loads(bind.read_text())
    b, name = Path(rec["board"]), rec["name"]
    lines = []
    seen_f = b / "cursor" / f"{name}.roster"
    try:
        seen = json.loads(seen_f.read_text())
    except (OSError, ValueError):
        seen = None
    cards = {r["name"]: r["card"] for r in mail.roster(b) if r["name"] != name}
    if seen is not None:  # join already printed the first roster
        lines += [f"peer {'updated' if k in seen else 'joined'} {k}: {v}" for k, v in cards.items() if seen.get(k) != v]
    seen_f.parent.mkdir(parents=True, exist_ok=True)
    seen_f.write_text(json.dumps(cards))
    lines += [mail.fmt(x) for x in mail.unread(b, name)]
    if not lines:
        return None
    _log(b, name, len(lines))
    return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                   "additionalContext": "mail for " + name + ":\n" + "\n".join(lines)}}


def _log(b: Path, name: str, n: int) -> None:
    import time
    with open(b / "pushes.jsonl", "a") as f:
        f.write(json.dumps({"name": name, "n": n, "ts": time.time()}) + "\n")


if __name__ == "__main__":
    try:
        out = run(json.load(sys.stdin))
    except Exception as e:  # noqa: BLE001 - fail open
        print(f"agent-mail hook: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(0)
    if out:
        print(json.dumps(out))
