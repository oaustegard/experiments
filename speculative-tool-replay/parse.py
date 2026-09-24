"""Parse Claude Code session transcripts into prediction points for speculative tool execution.

A prediction point is the moment after a tool result or user message when the model starts its
next assistant message. Its label is the set of read-only tool calls that message makes (possibly
empty). Timing comes from record timestamps:
  window = first tool_use timestamp - point timestamp  (model generation time before the call)
  dur    = tool_result timestamp - tool_use timestamp  (tool execution time)
Speculating at the point saves min(dur, window) for a call it predicted.
Sidechain (subagent) records are excluded.
"""
import json, re, shlex
from datetime import datetime
from pathlib import Path

READ_ONLY_TOOLS = {"Read", "Grep", "Glob", "WebFetch", "WebSearch", "mcp__Muninn__recall",
                   "mcp__Muninn__memory_get", "mcp__github__issue_read", "mcp__github__pull_request_read",
                   "mcp__github__list_issues", "mcp__github__list_pull_requests", "mcp__github__get_file_contents",
                   "ReadNotifications", "ListMcpResourcesTool", "ReadMcpResourceTool"}
# muninn_config is read-only only for get ops; handled in is_read_only.
RO_BASH = re.compile(r"^(ls|cat|head|tail|wc|grep|rg|find|du|df|stat|file|tree|pwd|which|env|printenv|date|"
                     r"git (status|log|diff|show|branch|rev-parse|ls-files|ls-tree|remote|config --get)|"
                     r"gh (pr|issue|run|api|repo) (view|list|diff|checks)?|jq|sed -n|awk|sort|uniq|echo)\b")
WRITE_HINT = re.compile(r"(>|>>|\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\btee\b|git (commit|push|add|checkout|reset|merge|rebase|fetch|pull|clone)|pip install|sed -i|\bcurl\b.*-X\s*(POST|PUT|DELETE)|nohup|&\s*$)")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def is_read_only(name, inp):
    if name in READ_ONLY_TOOLS:
        return True
    if name == "mcp__Muninn__muninn_config":
        return str(inp.get("op", inp.get("action", ""))).lower() in ("get", "list", "read", "")
    if name == "Bash":
        cmd = (inp.get("command") or "").strip()
        segs = re.split(r"\s*(?:&&|\|\||;|\n)\s*", cmd)
        return bool(cmd) and not WRITE_HINT.search(cmd) and all(RO_BASH.match(s) or not s for s in segs)
    return False


def key(name, inp):
    """Canonical identity of a call: what a speculative execution would have to match."""
    if name == "Read":
        return f"Read {inp.get('file_path')}"
    if name == "Bash":
        return "Bash " + " ".join((inp.get("command") or "").split())
    if name == "WebFetch":
        return f"WebFetch {inp.get('url')}"
    if name == "mcp__Muninn__memory_get":
        return f"memory_get {inp.get('id')}"
    return f"{name} " + json.dumps(inp, sort_keys=True)


def _text(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict):
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_result":
                c = b.get("content")
                out.append(c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict)))
    return "\n".join(out)


def events(path):
    """Linear non-sidechain event list: dicts with kind in user|text|call|result, t, and payload."""
    ev = []
    for line in open(path, errors="ignore"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("isSidechain") or r.get("type") not in ("user", "assistant") or not r.get("timestamp"):
            continue
        t, msg = ts(r["timestamp"]), r.get("message") or {}
        c = msg.get("content")
        if r["type"] == "user":
            blocks = [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    ev.append({"kind": "result", "t": t, "id": b.get("tool_use_id"), "text": _text([b])})
                elif b.get("type") == "text" and b.get("text", "").strip():
                    ev.append({"kind": "user", "t": t, "meta": bool(r.get("isMeta")) or b["text"].lstrip().startswith("<"),
                               "text": b["text"]})
        else:
            for b in c or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    ev.append({"kind": "text", "t": t, "mid": msg.get("id"), "text": b["text"]})
                elif b.get("type") == "tool_use":
                    ev.append({"kind": "call", "t": t, "mid": msg.get("id"), "id": b.get("id"),
                               "name": b.get("name"), "input": b.get("input") or {}})
    return ev


def points(path):
    """Prediction points with labels and timing."""
    ev = events(path)
    res_t = {e["id"]: e["t"] for e in ev if e["kind"] == "result"}
    out = []
    for i, e in enumerate(ev):
        if e["kind"] not in ("result", "user"):
            continue
        # next assistant message = run of text/call events sharing one message id
        j = i + 1
        while j < len(ev) and ev[j]["kind"] in ("result", "user"):
            j += 1  # parallel results / stacked user turns: the point is the last of them
        if j != i + 1 or j >= len(ev):
            continue
        mid = ev[j].get("mid")
        msg = [x for x in ev[j:] if x.get("mid") == mid] if mid else [ev[j]]
        calls = [x for x in msg if x["kind"] == "call"]
        ro = [c for c in calls if is_read_only(c["name"], c["input"])]
        first_t = calls[0]["t"] if calls else ev[j]["t"]
        labels = []
        for c in ro:
            dur = res_t.get(c["id"], c["t"]) - c["t"]
            labels.append({"key": key(c["name"], c["input"]), "name": c["name"], "dur": max(0.0, dur),
                           "window": max(0.0, c["t"] - e["t"])})
        out.append({"session": Path(path).name, "idx": i, "t": e["t"], "n_calls": len(calls),
                    "window": max(0.0, first_t - e["t"]), "labels": labels})
    return ev, out
