#!/usr/bin/env python3
"""Supervisor Stop hook: refuse to stop while an optimization run has budget
and is not exhausted (claude-workspace#233, AVO-style loop).

Inert unless an armed run config exists at .supervisor/run.json (or
$SUPERVISOR_RUN). The agent arms a run by writing the config and a ledger;
this hook then blocks each stop with a directive — next candidate, or
"change strategy class" on plateau — until budget is spent, the run is
exhausted, or a safety rail fires.

Safety rails (any one disarms the block, exit 0):
  - config missing / not armed / expired / malformed
  - kill file .supervisor/STOP next to the config
  - max_blocks_per_session exceeded (state file, per session_id)
  - two consecutive blocks with no new ledger entry (agent is stuck or
    refusing — stop harassing it)

Unlike check-store-on-stop.py this hook deliberately does NOT pass through
on stop_hook_active: repeated blocking is the mechanism. The rails above
are the loop bound instead.

On plateau, optionally asks Gemini (CF AI Gateway, ~5 s, creds inherited
from session env) for a strategy suggestion to include in the directive;
failures degrade to the plain arithmetic directive.

Exit codes: 0 pass, 2 block (stderr becomes the injected directive).
Always safe to crash: any exception → exit 0.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DEFAULT_RUN_PATH = ".supervisor/run.json"
GEMINI_TIMEOUT_S = 20


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _now() -> float:
    return time.time()


def _parse_iso(ts: str) -> float:
    """ISO-8601 UTC ('...Z' or offset) → epoch. Raises on garbage."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def plateaued(candidates: list[dict], window: int, min_delta: float) -> bool:
    """True when the last `window` candidates all failed to beat the best
    score seen before them by at least min_delta."""
    if len(candidates) <= window:
        return False
    scores = [float(c.get("score", float("-inf"))) for c in candidates]
    best_before = max(scores[:-window])
    return all(s < best_before + min_delta for s in scores[-window:])


def decide(run: dict, ledger: dict, state: dict, now: float) -> tuple[int, str, dict]:
    """Pure decision core. Returns (exit_code, directive, new_state).

    run: parsed run.json; ledger: parsed ledger.json; state: supervisor's own
    per-session state {blocks, last_count, no_progress_blocks}.
    """
    state = dict(state)

    if not run.get("armed"):
        return 0, "", state
    try:
        if now > _parse_iso(run["expires_at"]):
            return 0, "", state
    except Exception:
        return 0, "", state  # unparseable expiry = expired, fail safe

    candidates = ledger.get("candidates", [])
    n = len(candidates)
    max_candidates = int(run.get("budget", {}).get("max_candidates", 0))
    max_blocks = int(run.get("max_blocks_per_session", 10))

    # Safety: blocks-per-session bound
    if state.get("blocks", 0) >= max_blocks:
        return 0, "", state

    # Safety: no progress across two consecutive blocks → agent is stuck
    if n == state.get("last_count", -1):
        state["no_progress_blocks"] = state.get("no_progress_blocks", 0) + 1
        if state["no_progress_blocks"] >= 2:
            return 0, "", state
    else:
        state["no_progress_blocks"] = 0
    state["last_count"] = n

    # Budget spent → let the stop through
    if n >= max_candidates:
        return 0, "", state

    scores = [float(c.get("score", float("-inf"))) for c in candidates]
    best = max(scores) if scores else None
    best_txt = f"{best:.4f}" if best is not None else "n/a"

    plat = run.get("plateau", {})
    window = int(plat.get("window", 3))
    min_delta = float(plat.get("min_delta", 0.0))
    max_switches = int(plat.get("max_strategy_switches", 2))
    switches = int(ledger.get("strategy_switches", 0))

    state["blocks"] = state.get("blocks", 0) + 1

    if plateaued(candidates, window, min_delta):
        if switches >= max_switches:
            return 0, "", state  # run exhausted: plateaued in every allowed class
        msg = (
            f"SUPERVISOR ({run.get('run_id', 'run')}): plateau — last {window} "
            f"candidates did not improve on best {best_txt} by {min_delta}. "
            f"CHANGE STRATEGY CLASS (switch {switches + 1} of {max_switches}). "
            f"Increment strategy_switches in the ledger, then run the next "
            f"candidate from a different class, record it, and stop again. "
            f"Goal: {run.get('goal', '')}"
        )
        return 2, msg, state

    msg = (
        f"SUPERVISOR ({run.get('run_id', 'run')}): budget remains — "
        f"{n} of {max_candidates} candidates done, best {best_txt}. "
        f"Run the next candidate ({run.get('fitness_cmd', 'fitness')}), append "
        f"it to the ledger with a one-line description and strategy_class, "
        f"then stop again. Do not summarize; do not ask. "
        f"Goal: {run.get('goal', '')}"
    )
    return 2, msg, state


def _gemini_suggestion(run: dict, ledger: dict) -> str:
    """Best-effort tier-2 escalation. Returns '' on any failure."""
    try:
        sys.path.insert(0, "/mnt/skills/user/invoking-gemini/scripts")
        from gemini_client import invoke_gemini  # noqa: PLC0415

        tail = ledger.get("candidates", [])[-8:]
        prompt = (
            "An optimization loop plateaued. Goal: "
            + str(run.get("goal", ""))
            + "\nRecent candidates (desc, strategy_class, score): "
            + json.dumps(
                [
                    {k: c.get(k) for k in ("desc", "strategy_class", "score")}
                    for c in tail
                ]
            )
            + "\nIn two sentences: which strategy class to try next and why."
        )
        out = invoke_gemini(prompt, model="gemini-3.6-flash")
        return str(out).strip()[:600]
    except Exception:
        return ""


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    try:
        run_path = Path(os.environ.get("SUPERVISOR_RUN", DEFAULT_RUN_PATH))
        if not run_path.is_absolute():
            run_path = Path(data.get("cwd", ".")) / run_path
        run = _read_json(run_path)
        if not run:
            return 0
        base = run_path.parent
        if (base / "STOP").exists():
            return 0

        ledger_path = Path(run.get("ledger", "ledger.json"))
        if not ledger_path.is_absolute():
            ledger_path = base / ledger_path
        ledger = _read_json(ledger_path) or {"candidates": []}

        session = str(data.get("session_id", "unknown"))[:16]
        state_path = base / f"state-{session}.json"
        state = _read_json(state_path) or {}

        code, msg, new_state = decide(run, ledger, state, _now())

        # Observability: every invocation logged next to the run
        try:
            with (base / "supervisor-log.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": _now(),
                            "stop_hook_active": data.get("stop_hook_active"),
                            "exit": code,
                            "candidates": len(ledger.get("candidates", [])),
                            "argv": sys.argv[1:],
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

        if code == 2 and "CHANGE STRATEGY CLASS" in msg and run.get("gemini_escalation"):
            hint = _gemini_suggestion(run, ledger)
            if hint:
                msg += f"\nGemini suggestion: {hint}"

        try:
            state_path.write_text(json.dumps(new_state))
        except Exception:
            pass

        if code == 2:
            sys.stderr.write(msg + "\n")
            return 2
        return 0
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(main())
