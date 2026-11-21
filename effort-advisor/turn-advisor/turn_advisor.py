#!/usr/bin/env python3
"""turn_advisor: per-turn effort and delegation advice with no added latency.

On every human prompt a detached child asks Haiku (one low-effort call, prompt
only) what the coming turn deserves. The prompt is never held up: the advice is
delivered at the first tool boundary after it is ready, as additionalContext.
A turn that makes no tool call never needed the advice.

What it can do, and why only this:
  * RAISE EFFORT for the rest of the turn, by telling the model to invoke the
    escalating-effort skill (frontmatter `effort: xhigh`). Effort is a request
    parameter, so the prompt cache survives. It cannot lower effort and it
    cannot switch model: a model switch after the first turn re-reads the whole
    context at full price (see model_advisor.py for the first-prompt case).
  * NUDGE DELEGATION when the turn has a separable search/read chunk.
It stays silent otherwise, and logs every decision either way.

Wiring (.claude/settings.json), all three are the same script:
    UserPromptSubmit  python3 scripts/turn_advisor.py prompt
    PostToolUse (*)   python3 scripts/turn_advisor.py deliver PostToolUse
    PreToolUse  (*)   python3 scripts/turn_advisor.py deliver PreToolUse   (optional: earlier delivery)

    python3 turn_advisor.py classify "refactor the auth module"   # one-off, prints the decision
    python3 turn_advisor.py report                                # fire rate, latency, delivery rate
    TURN_ADVISOR=off disables; TURN_ADVISOR_FAKE='xhigh:0.9:none' stubs the classifier (tests).

Fail open everywhere: any error means no advice, and a line in the log.
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

HOME = Path(os.environ.get("TURN_ADVISOR_HOME") or Path(__file__).resolve().parent / ".turn-advisor")
STATE, LOG = HOME / "state", HOME / "turn-advisor.jsonl"
MODEL = os.environ.get("TURN_ADVISOR_MODEL", "claude-haiku-5-5")
TIMEOUT = float(os.environ.get("TURN_ADVISOR_TIMEOUT", "30"))
MIN_CONF = float(os.environ.get("TURN_ADVISOR_MIN_CONF", "0.7"))   # escalate only when Haiku is this sure
ESCALATE_AT = os.environ.get("TURN_ADVISOR_ESCALATE_AT", "xhigh")  # "high" fires more often; see RESULTS.md
SKILL = os.environ.get("TURN_ADVISOR_SKILL", "escalating-effort")
# shadow (default): classify and log, inject nothing. act: deliver the advice. The 2026-11-15 study (RESULTS.md)
# did not show Haiku picking the right turns to escalate, so it ships logging only.
MODE = os.environ.get("TURN_ADVISOR_MODE", "shadow")
EFFORTS = ["low", "medium", "high", "xhigh"]
SKIP_ENTRYPOINTS = re.compile(r"^(remote_trigger|sdk)")
OPT_OUT = "#no-advice"
PROMPT_CHARS, CONTEXT_CHARS = 3000, 700

SYSTEM = """You label one user message sent to an AI coding agent. Judge what the coming turn of work deserves, from the message and the short context alone. Reply with one JSON object and nothing else.

effort = the least reasoning effort at which a capable model does this turn well:
- low: acknowledgement, go-ahead, a named command to run, a simple lookup, a short factual or conversational answer, a mechanical edit the message spells out.
- medium: routine focused work with a clear deliverable: implement or fix a described change, summarise or review one thing, a standard multi-step task with no hard judgment.
- high: design judgment, debugging an unknown cause, careful analysis or critique, weighing trade-offs, a plan others will build on.
- xhigh: hard reasoning where a subtle mistake is costly: research or experiment design, novel algorithms, subtle correctness or security, long autonomous projects with many dependent decisions.

delegate = should the agent hand part of the turn to subagents?
- none: do it inline.
- subagent: one separable chunk (a broad search, reading many files, an independent check).
- fanout: several independent pieces worth running in parallel.

A short message such as "yes" or "do it" takes its meaning from the previous assistant reply: judge the work it sets off.

{"effort": "low|medium|high|xhigh", "conf": <0-1, how sure of effort>, "delegate": "none|subagent|fanout"}"""


def log(rec: dict) -> None:
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")
    except OSError:
        pass


def transcript_tail(path: str | None) -> dict:
    """Last assistant text, model and effort from the session transcript (absent on a first prompt)."""
    out = {"reply": "", "model": None, "effort": None}
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 400_000))
            lines = f.read().decode(errors="ignore").splitlines()[1 if size > 400_000 else 0:]
    except (OSError, TypeError):
        return out
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != "assistant" or d.get("isSidechain"):
            continue
        m = d.get("message") or {}
        out["model"] = out["model"] or m.get("model")
        out["effort"] = out["effort"] or d.get("perTurnEffort") or d.get("effort")
        if not out["reply"]:
            out["reply"] = "\n".join(b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text").strip()
        if out["reply"] and out["model"]:
            break
    return out


def ask_haiku(prompt: str, prev_reply: str) -> dict:
    fake = os.environ.get("TURN_ADVISOR_FAKE")
    if fake:
        if fake == "error":
            raise RuntimeError("fake classifier error")
        e, c, d = (fake.split(":") + ["0.9", "none"])[:3]
        return {"effort": e, "conf": float(c), "delegate": d}
    user = json.dumps({"previous_assistant_reply_tail": prev_reply[-CONTEXT_CHARS:], "user_message": prompt[:PROMPT_CHARS]}, ensure_ascii=False)
    argv = ["claude", "-p", "--model", MODEL, "--effort", "low", "--output-format", "json", "--system-prompt", SYSTEM,
            "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence"]
    env = dict(os.environ, TURN_ADVISOR="off")  # belt and braces: the child loads no settings, so no hooks
    r = subprocess.run(argv, input=user, capture_output=True, text=True, timeout=TIMEOUT, env=env)
    out = json.loads(r.stdout)
    if out.get("is_error"):
        raise RuntimeError(str(out.get("result"))[:200])
    m = re.search(r"\{.*\}", out.get("result") or "", re.S)
    ans = json.loads(m.group(0))
    if ans.get("effort") not in EFFORTS:
        raise ValueError(f"bad effort {ans.get('effort')!r}")
    return {"effort": ans["effort"], "conf": float(ans.get("conf") or 0), "delegate": ans.get("delegate") or "none"}


def advice_for(ans: dict, current_effort: str | None) -> tuple[str | None, list[str]]:
    """The note to inject (or None) and the rules that fired."""
    notes, rules = [], []
    cur = EFFORTS.index(current_effort) if current_effort in EFFORTS else 1   # unknown: assume medium
    need = EFFORTS.index(ans["effort"])
    if need >= EFFORTS.index(ESCALATE_AT) and need > cur and ans["conf"] >= MIN_CONF and cur < EFFORTS.index("xhigh"):
        rules.append("escalate")
        notes.append(f"this turn looks like {ans['effort']}-effort work and the session is at {current_effort or 'its default'}. "
                     f"Invoke the Skill tool with skill `{SKILL}` now, then continue the task.")
    if ans["delegate"] in ("subagent", "fanout"):
        rules.append("delegate")
        notes.append("part of this turn is separable (a broad search, many files to read, or independent pieces). "
                     + ("Run the independent pieces as parallel subagents" if ans["delegate"] == "fanout" else "Hand that chunk to a subagent")
                     + " if you have not already, and keep the conclusion, not the file dumps.")
    if not notes:
        return None, rules
    return "Turn advisor (automatic, from a Haiku read of the user's message; the user has not seen this): " + " Also: ".join(notes), rules


def cmd_prompt() -> None:
    d = json.load(sys.stdin)
    sid, prompt = d.get("session_id") or "nosession", d.get("prompt") or ""
    STATE.mkdir(parents=True, exist_ok=True)
    for p in STATE.glob(sid + ".*"):          # a new turn voids advice left over from the last one
        p.unlink(missing_ok=True)
    skip = ("off" if os.environ.get("TURN_ADVISOR", "").lower() == "off" else
            "entrypoint" if SKIP_ENTRYPOINTS.match(os.environ.get("CLAUDE_CODE_ENTRYPOINT", "")) else
            "opt-out" if OPT_OUT in prompt else
            "command" if prompt.lstrip().startswith(("/", "<")) else None)
    if skip:
        log({"event": "skip", "session": sid[:8], "why": skip}); return
    req = STATE / (sid + ".request")
    req.write_text(json.dumps({"session": sid, "prompt": prompt, "transcript": d.get("transcript_path"), "t0": time.time(),
                               "env_effort": os.environ.get("CLAUDE_EFFORT")}))
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", str(req)], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def cmd_worker(req_path: str) -> None:
    req = json.loads(Path(req_path).read_text())
    sid = req["session"]
    tail = transcript_tail(req.get("transcript"))
    current = tail["effort"] or req.get("env_effort")
    rec = {"event": "decision", "session": sid[:8], "chars": len(req["prompt"]), "model": tail["model"], "effort_now": current}
    try:
        ans = ask_haiku(req["prompt"], tail["reply"])
        note, rules = advice_for(ans, current)
        rec.update(ans, rules=rules, mode=MODE, ms=round((time.time() - req["t0"]) * 1000))
        if note and MODE == "act" and Path(req_path).exists():   # the request file is gone if a newer prompt arrived meanwhile
            tmp = STATE / (sid + ".advice.tmp")
            tmp.write_text(note); os.replace(tmp, STATE / (sid + ".advice"))
    except Exception as e:  # fail open
        rec.update(error=f"{type(e).__name__}: {str(e)[:200]}", ms=round((time.time() - req["t0"]) * 1000))
    log(rec)


def cmd_deliver(event: str) -> None:
    d = json.load(sys.stdin)
    sid = d.get("session_id") or "nosession"
    f = STATE / (sid + ".advice")
    if not f.exists():
        return
    done = STATE / (sid + ".delivered")
    try:
        os.rename(f, done)                     # atomic claim: parallel tool calls deliver once
    except OSError:
        return
    note = done.read_text()
    log({"event": "delivered", "session": sid[:8], "via": event, "tool": d.get("tool_name")})
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": note}}))


def cmd_classify(text: str) -> None:
    t = time.time()
    ans = ask_haiku(text, "")
    note, rules = advice_for(ans, os.environ.get("CLAUDE_EFFORT"))
    print(json.dumps({**ans, "rules": rules, "ms": round((time.time() - t) * 1000), "note": note}, indent=1))


def cmd_report() -> None:
    rows = [json.loads(l) for l in open(LOG)] if LOG.exists() else []
    dec = [r for r in rows if r["event"] == "decision"]
    ok = [r for r in dec if "error" not in r]
    ms = sorted(r["ms"] for r in ok)
    print(f"prompts classified {len(ok)}  errors {len(dec) - len(ok)}  skipped {sum(r['event'] == 'skip' for r in rows)}")
    if ms:
        print(f"latency ms: median {ms[len(ms) // 2]}  p90 {ms[int(len(ms) * .9)]}")
    for rule in ("escalate", "delegate"):
        print(f"{rule}: fired {sum(rule in r.get('rules', []) for r in ok)}")
    print(f"advice delivered {sum(r['event'] == 'delivered' for r in rows)}")
    from collections import Counter
    print("effort called:", dict(Counter(r["effort"] for r in ok)))


if __name__ == "__main__":
    try:
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd == "prompt": cmd_prompt()
        elif cmd == "worker": cmd_worker(sys.argv[2])
        elif cmd == "deliver": cmd_deliver(sys.argv[2] if len(sys.argv) > 2 else "PostToolUse")
        elif cmd == "classify": cmd_classify(sys.argv[2])
        elif cmd == "report": cmd_report()
        else: print(__doc__)
    except Exception as e:  # a hook must never break the session
        log({"event": "crash", "cmd": sys.argv[1:2], "error": f"{type(e).__name__}: {str(e)[:200]}"})
    sys.exit(0)
