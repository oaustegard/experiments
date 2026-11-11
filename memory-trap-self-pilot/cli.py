"""claude -p as a model client: isolated (no MCP, skills, settings, tools), optional --resume for multi-turn."""
import json, subprocess, time, os
CWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run")
BASE = ["claude", "-p", "--tools", "", "--output-format", "json", "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--no-chrome", "--setting-sources", ""]

def call(prompt, model, system, resume=None, persist=False, tries=4, timeout=240):
    cmd = BASE + ["--model", model, "--system-prompt", system]
    if resume: cmd += ["--resume", resume]
    if not persist and not resume: cmd += ["--no-session-persistence"]
    last = None
    for i in range(tries):
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=CWD)
            j = json.loads(p.stdout)
            if j.get("is_error"): raise RuntimeError(str(j.get("result"))[:200])
            return j.get("result") or "", j.get("session_id"), j.get("usage", {}), j.get("total_cost_usd", 0.0)
        except Exception as e:
            last = e; time.sleep(3 * (i + 1))
    raise RuntimeError(f"claude -p failed: {last}")
