"""Shared helpers for the hooktest hooks. Every hook logs its input to logs/<event>.jsonl."""
import json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LOGS, STATE = ROOT / "logs", ROOT / "state"
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTH|SESSION_ACCESS", re.I)

def read_input():
    try:
        return json.load(sys.stdin)
    except ValueError:
        return {}

def redacted_env():
    return {k: ("<redacted %d chars>" % len(v) if SECRET.search(k) else v[:300]) for k, v in sorted(os.environ.items())}

def log(event, data, env=False):
    LOGS.mkdir(exist_ok=True)
    rec = {"t": time.time(), "mode": os.environ.get("HOOKTEST_MODE"), "input": data}
    if env:
        rec["env"] = redacted_env()
    with open(LOGS / f"{event}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

def emit_context(event, text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))
