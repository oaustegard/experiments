"""Candidate next calls at a prediction point, drawn only from what precedes it.

Jev selects; it cannot write a call. So a call is predictable only if code can put it on the menu:
file paths, URLs, memory ids and config keys that appear earlier in the session, plus exact repeats
of earlier read-only calls. Candidates are ordered most-recent-first.
"""
import re
from parse import is_read_only, key

PATH = re.compile(r"(?<![\w.:/-])(/(?:home|root|mnt|tmp|opt|etc|workspace|usr)/[\w.@+/-]+\.[A-Za-z0-9]{1,6})(?![\w/])")
URL = re.compile(r"https?://[^\s\"'<>)\]`]+")
HEX8 = re.compile(r"\b([0-9a-f]{8})\b")
CFG = re.compile(r"config_get\(['\"]([\w-]+)['\"]\)")
MENU = ["Bash git status", "Bash git log --oneline -5", "Bash git diff --stat", "Bash ls"]


def _strings(e):
    if e["kind"] == "call":
        return [str(v) for v in e["input"].values()]
    return [e.get("text", "")]


def candidates(ev, idx, limit=None):
    """Candidate keys for the point at ev[idx], most recent first, deduplicated."""
    seen, out = set(), []

    def add(k):
        if k not in seen:
            seen.add(k); out.append(k)

    for e in reversed(ev[: idx + 1]):
        if e["kind"] == "call" and is_read_only(e["name"], e["input"]):
            add(key(e["name"], e["input"]))
        for s in _strings(e):
            s = s[:20000]
            for p in PATH.findall(s):
                add(f"Read {p}")
            for u in URL.findall(s):
                add(f"WebFetch {u.rstrip('.,;')}")
            for h in HEX8.findall(s):
                if not h.isdigit() and re.search(r"[a-f]", h):
                    add(f"memory_get {h}")
            for c in CFG.findall(s):
                add(f'mcp__Muninn__muninn_config {{"key": "{c}", "op": "get"}}')
        if limit and len(out) >= limit:
            break
    for m in MENU:
        add(m)
    return out[:limit] if limit else out
