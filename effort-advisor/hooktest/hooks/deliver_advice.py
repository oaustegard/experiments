#!/usr/bin/env python3
"""PreToolUse / PostToolUse hook: deliver state/<session>.advice once, as additionalContext.
argv[1] is the event name. HOOKTEST_DELIVER = pre | post | both (default post) limits which event delivers."""
import os, sys
from common import STATE, emit_context, log, read_input
event = sys.argv[1]
d = read_input()
log(event, d)
want = os.environ.get("HOOKTEST_DELIVER", "post")
if os.environ.get("HOOKTEST_MODE") != "async" or want not in ("both", {"PreToolUse": "pre", "PostToolUse": "post"}[event]):
    sys.exit(0)
f = STATE / (d.get("session_id", "nosession") + ".advice")
done = f.with_suffix(".delivered")
if f.exists() and not done.exists():
    try:
        os.rename(f, done)  # atomic claim: parallel tool calls deliver only once
    except OSError:
        sys.exit(0)
    log("delivered", {"via": event, "tool": d.get("tool_name"), "tool_use_id": d.get("tool_use_id")})
    emit_context(event, done.read_text())
