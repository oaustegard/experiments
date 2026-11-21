"""Minimal headless Claude call: no tools, no settings, no MCP, no session file."""
import json, subprocess, time, re
def call(model, system, user, effort=None, timeout=300):
    argv = ["claude", "-p", "--model", model, "--output-format", "json", "--system-prompt", system,
            "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence"]
    if effort: argv += ["--effort", effort]
    t = time.time()
    try:
        r = subprocess.run(argv, input=user, capture_output=True, text=True, timeout=timeout)
        out = json.loads(r.stdout)
    except Exception as e:
        return None, {"error": str(e)[:200], "s": time.time() - t}
    meta = {"s": time.time() - t, "cost": out.get("total_cost_usd"), "api_ms": out.get("duration_api_ms"), "err": out.get("is_error")}
    return out.get("result"), meta
def parse_json(text):
    if not text: return None
    m = re.search(r'(\{.*\}|\[.*\])', text, re.S)
    if not m: return None
    try: return json.loads(m.group(1))
    except Exception: return None
