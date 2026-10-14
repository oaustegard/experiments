#!/usr/bin/env python3
"""Agent mailbox: one append-only log, per-reader cursors, a roster of cards.

Peers in a swarm address each other by name. The board is a directory:

    roster.jsonl   one card per agent: {"name", "card", "ts"}
    log.jsonl      every message: {"seq", "from", "to", "text", "ts"}
    cursor/<name>  last seq that <name> has seen
    bind/<agent_id>  name bound to a Claude Code agent_id (written by hook.py)

Commands (board dir from $AGENT_MAIL_DIR, else ~/.claude/agent-mail/ACTIVE):

    mail.py join NAME "CARD"          register; prints the roster and any unread mail
    mail.py send FROM TO "TEXT"       TO is a name, a comma list, or "all"
    mail.py recv NAME [--wait SECS]   print unread mail; --wait blocks until some arrives
    mail.py roster                    print every card
    mail.py log                       dump the whole log (for the orchestrator)

Output is one line per item, "[#seq from>to] text", so a reader pays for the
message and nothing else.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

ACTIVE = Path.home() / ".claude" / "agent-mail" / "ACTIVE"


def board() -> Path | None:
    d = os.environ.get("AGENT_MAIL_DIR")
    if not d and ACTIVE.exists():
        d = ACTIVE.read_text().strip()
    return Path(d) if d else None


def _append(path: Path, rec: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if "seq" in rec:
            f.seek(0)
            rec["seq"] = sum(1 for _ in f) + 1
        f.write(json.dumps(rec) + "\n")
    return rec


def _read(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def roster(b: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in _read(b / "roster.jsonl"):
        latest[r["name"]] = r
    return list(latest.values())


def fmt(m: dict) -> str:
    return f"[#{m['seq']} {m['from']}>{m['to']}] {m['text']}"


def unread(b: Path, name: str, advance: bool = True) -> list[dict]:
    cur_f = b / "cursor" / name
    try:
        cur = int(cur_f.read_text())
    except (OSError, ValueError):
        cur = 0
    msgs = [m for m in _read(b / "log.jsonl") if m["seq"] > cur and m["from"] != name
            and (m["to"] == "all" or name in m["to"].split(","))]
    if advance:
        top = max([cur] + [m["seq"] for m in _read(b / "log.jsonl")])
        cur_f.parent.mkdir(parents=True, exist_ok=True)
        cur_f.write_text(str(top))
    return msgs


def main(argv: list[str]) -> int:
    b = board()
    if b is None:
        print("mail: no board (set AGENT_MAIL_DIR or ~/.claude/agent-mail/ACTIVE)", file=sys.stderr)
        return 2
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 64
    cmd, args = argv[1], argv[2:]
    if cmd == "join" and len(args) == 2:
        _append(b / "roster.jsonl", {"name": args[0], "card": args[1], "ts": time.time()})
        for r in roster(b):
            if r["name"] != args[0]:
                print(f"{r['name']}: {r['card']}")
        for m in unread(b, args[0]):
            print(fmt(m))
        return 0
    if cmd == "send" and len(args) == 3:
        m = _append(b / "log.jsonl", {"seq": 0, "from": args[0], "to": args[1], "text": args[2], "ts": time.time()})
        print(f"sent #{m['seq']}")
        return 0
    if cmd == "recv" and args:
        wait = float(args[2]) if len(args) == 3 and args[1] == "--wait" else 0.0
        deadline = time.time() + wait
        while True:
            msgs = unread(b, args[0])
            if msgs or time.time() >= deadline:
                break
            time.sleep(1)
        print("\n".join(fmt(m) for m in msgs) if msgs else "(no mail)")
        return 0
    if cmd == "roster":
        for r in roster(b):
            print(f"{r['name']}: {r['card']}")
        return 0
    if cmd == "log":
        for m in _read(b / "log.jsonl"):
            print(fmt(m))
        return 0
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
