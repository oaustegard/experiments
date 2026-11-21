"""Offline tests: stubbed classifier, real hook plumbing (spawn, atomic delivery, skips)."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
S = str(Path(__file__).resolve().parents[1] / "turn_advisor.py")

def run(cmd, payload, home, **env):
    e = dict(os.environ, TURN_ADVISOR_HOME=home, CLAUDE_CODE_ENTRYPOINT="cli", TURN_ADVISOR_MODE="act"); e.pop("CLAUDE_EFFORT", None); e.pop("TURN_ADVISOR", None)
    e.update(env)
    return subprocess.run([sys.executable, S, *cmd], input=json.dumps(payload), capture_output=True, text=True, env=e)

def wait_log(home, event, n=1, secs=5):
    f = Path(home) / "turn-advisor.jsonl"
    for _ in range(int(secs * 20)):
        rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        hit = [r for r in rows if r["event"] == event]
        if len(hit) >= n: return hit
        time.sleep(0.05)
    raise AssertionError(f"no {event} in log: {f.read_text() if f.exists() else 'no log'}")

def transcript(tmp, effort):
    p = Path(tmp) / "t.jsonl"
    p.write_text(json.dumps({"type": "assistant", "perTurnEffort": effort, "message": {"model": "claude-opus-5-5", "content": [{"type": "text", "text": "Shall I go ahead?"}]}}) + "\n")
    return str(p)

def test_escalates_and_delivers_once():
    with tempfile.TemporaryDirectory() as h:
        p = {"session_id": "s1", "prompt": "design the experiment", "transcript_path": transcript(h, "medium")}
        r = run(["prompt"], p, h, TURN_ADVISOR_FAKE="xhigh:0.9:none"); assert r.stdout == ""   # prompt is never held up
        d = wait_log(h, "decision")[0]; assert d["rules"] == ["escalate"] and d["effort_now"] == "medium"
        out = run(["deliver", "PostToolUse"], {"session_id": "s1", "tool_name": "Bash"}, h).stdout
        ctx = json.loads(out)["hookSpecificOutput"]; assert ctx["hookEventName"] == "PostToolUse" and "escalating-effort" in ctx["additionalContext"]
        assert run(["deliver", "PostToolUse"], {"session_id": "s1"}, h).stdout == ""           # once only

def test_silent_when_already_high_or_unsure_or_low():
    for fake, eff in (("xhigh:0.9:none", "xhigh"), ("xhigh:0.5:none", "medium"), ("medium:0.9:none", "low"), ("high:0.9:none", "medium")):
        with tempfile.TemporaryDirectory() as h:
            run(["prompt"], {"session_id": "s2", "prompt": "x y z", "transcript_path": transcript(h, eff)}, h, TURN_ADVISOR_FAKE=fake)
            assert wait_log(h, "decision")[0]["rules"] == [], (fake, eff)
            assert run(["deliver", "PostToolUse"], {"session_id": "s2"}, h).stdout == ""

def test_delegate_note():
    with tempfile.TemporaryDirectory() as h:
        run(["prompt"], {"session_id": "s3", "prompt": "audit every repo", "transcript_path": None}, h, TURN_ADVISOR_FAKE="medium:0.9:fanout")
        assert wait_log(h, "decision")[0]["rules"] == ["delegate"]
        assert "parallel subagents" in run(["deliver", "PreToolUse"], {"session_id": "s3"}, h).stdout

def test_skips_and_fail_open():
    with tempfile.TemporaryDirectory() as h:
        run(["prompt"], {"session_id": "s4", "prompt": "#no-advice do x"}, h, TURN_ADVISOR_FAKE="xhigh")
        run(["prompt"], {"session_id": "s4", "prompt": "do x"}, h, TURN_ADVISOR_FAKE="xhigh", CLAUDE_CODE_ENTRYPOINT="remote_trigger")
        run(["prompt"], {"session_id": "s4", "prompt": "do x"}, h, TURN_ADVISOR="off")
        assert [r["why"] for r in wait_log(h, "skip", 3)] == ["opt-out", "entrypoint", "off"]
        run(["prompt"], {"session_id": "s5", "prompt": "do x"}, h, TURN_ADVISOR_FAKE="error")
        assert "error" in wait_log(h, "decision")[0]
        assert run(["deliver", "PostToolUse"], {"session_id": "s5"}, h).stdout == ""
        assert run(["deliver", "PostToolUse"], "not json", h).returncode == 0

def test_new_prompt_voids_old_advice():
    with tempfile.TemporaryDirectory() as h:
        run(["prompt"], {"session_id": "s6", "prompt": "hard thing"}, h, TURN_ADVISOR_FAKE="xhigh:0.9:none"); wait_log(h, "decision")
        run(["prompt"], {"session_id": "s6", "prompt": "thanks"}, h, TURN_ADVISOR_FAKE="low:0.9:none"); wait_log(h, "decision", 2)
        assert run(["deliver", "PostToolUse"], {"session_id": "s6"}, h).stdout == ""

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)

def test_shadow_mode_logs_but_injects_nothing():
    with tempfile.TemporaryDirectory() as h:
        run(["prompt"], {"session_id": "s7", "prompt": "design it"}, h, TURN_ADVISOR_FAKE="xhigh:0.9:fanout", TURN_ADVISOR_MODE="shadow")
        assert wait_log(h, "decision")[0]["rules"] == ["escalate", "delegate"]
        assert run(["deliver", "PostToolUse"], {"session_id": "s7"}, h).stdout == ""
test_shadow_mode_logs_but_injects_nothing(); print("ok shadow")
