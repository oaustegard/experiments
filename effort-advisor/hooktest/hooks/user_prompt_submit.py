#!/usr/bin/env python3
"""UserPromptSubmit hook. HOOKTEST_MODE picks the experiment:
  nonce     inject a nonce through additionalContext
  escalate  tell the model to invoke the skill named in HOOKTEST_SKILL
  async     return at once; a detached child writes state/<session>.advice after HOOKTEST_DELAY s
  delegate  inject the Explore-subagent nudge
  recurse   call `claude -p` from inside the hook with HOOKTEST_INNER_FLAGS (stripped|bare)
"""
import os, subprocess, sys, time
from common import STATE, emit_context, log, read_input
d = read_input()
mode = os.environ.get("HOOKTEST_MODE", "")
log("UserPromptSubmit", d, env=True)
EV = "UserPromptSubmit"
if mode == "nonce":
    emit_context(EV, "The advisor nonce is %s. If asked for the advisor nonce, state it." % os.environ.get("HOOKTEST_NONCE", "ZEBRA-4471"))
elif mode == "escalate":
    emit_context(EV, "Before anything else, invoke the Skill tool with skill %s. Then do the user's task." % os.environ.get("HOOKTEST_SKILL", "escalating-effort"))
elif mode == "async":
    STATE.mkdir(exist_ok=True)
    f = STATE / (d.get("session_id", "nosession") + ".advice")
    for p in (f, f.with_suffix(".delivered")):
        p.unlink(missing_ok=True)
    delay = os.environ.get("HOOKTEST_DELAY", "3")
    nonce = os.environ.get("HOOKTEST_NONCE", "HERON-9024")
    # stand-in for the Haiku classifier: sleep, then write the advice atomically
    subprocess.Popen(["/bin/sh", "-c", 'sleep %s; printf "%%s" "Turn advisor: the advisor nonce is %s. Mention it in your final answer." > "%s.tmp" && mv "%s.tmp" "%s"' % (delay, nonce, f, f, f)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
elif mode == "delegate":
    emit_context(EV, "This task has an independent search component; delegate it to an Explore subagent via the Agent tool.")
elif mode == "recurse":
    flags = {"stripped": ["--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence"],
             "bare": []}[os.environ.get("HOOKTEST_INNER_FLAGS", "stripped")]
    env = dict(os.environ, HOOKTEST_MODE="inner")  # so a recursive hook run is visible in the log, and cannot recurse again
    t = time.time()
    r = subprocess.run(["claude", "-p", "--model", "haiku", "--effort", "low", *flags, "Reply with the single word pong."],
                       capture_output=True, text=True, env=env, timeout=90, cwd=os.environ.get("HOOKTEST_INNER_CWD") or None)
    log("inner_call", {"flags": flags, "rc": r.returncode, "out": r.stdout[-300:], "err": r.stderr[-300:], "secs": round(time.time() - t, 2)})
sys.exit(0)
