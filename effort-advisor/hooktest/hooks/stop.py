#!/usr/bin/env python3
"""Stop hook. With HOOKTEST_STOP=block and advice still undelivered, wait up to HOOKTEST_STOP_WAIT s for the
advice file, then block the stop once with the advice as the reason (the model gets another request)."""
import json, os, sys, time
from common import STATE, log, read_input
d = read_input()
log("Stop", d, env=True)
if os.environ.get("HOOKTEST_MODE") != "async" or os.environ.get("HOOKTEST_STOP") != "block" or d.get("stop_hook_active"):
    sys.exit(0)
f = STATE / (d.get("session_id", "nosession") + ".advice")
done = f.with_suffix(".delivered")
end = time.time() + float(os.environ.get("HOOKTEST_STOP_WAIT", "5"))
while not f.exists() and not done.exists() and time.time() < end:
    time.sleep(0.2)
if f.exists() and not done.exists():
    os.rename(f, done)
    log("delivered", {"via": "Stop"})
    print(json.dumps({"decision": "block", "reason": done.read_text()}))
