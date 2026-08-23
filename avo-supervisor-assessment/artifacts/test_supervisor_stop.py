"""Tests for the pure decision core of scripts/supervisor_stop.py."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "supervisor_stop.py"

spec = importlib.util.spec_from_file_location("supervisor_stop", SCRIPT)
sup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sup)

NOW = time.time()
FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


def run_cfg(**over):
    cfg = {
        "armed": True,
        "run_id": "t",
        "expires_at": FUTURE,
        "budget": {"max_candidates": 5},
        "plateau": {"window": 2, "min_delta": 0.001, "max_strategy_switches": 2},
        "max_blocks_per_session": 10,
        "goal": "g",
    }
    cfg.update(over)
    return cfg


def led(scores, switches=0):
    return {
        "candidates": [
            {"id": i, "score": s, "desc": "d", "strategy_class": "a"}
            for i, s in enumerate(scores)
        ],
        "strategy_switches": switches,
    }


def test_unarmed_passes():
    code, _, _ = sup.decide(run_cfg(armed=False), led([]), {}, NOW)
    assert code == 0


def test_expired_passes():
    code, _, _ = sup.decide(run_cfg(expires_at=PAST), led([]), {}, NOW)
    assert code == 0


def test_unparseable_expiry_passes():
    code, _, _ = sup.decide(run_cfg(expires_at="soon"), led([]), {}, NOW)
    assert code == 0


def test_budget_remaining_blocks_with_directive():
    code, msg, state = sup.decide(run_cfg(), led([0.7]), {}, NOW)
    assert code == 2
    assert "1 of 5" in msg and "0.7000" in msg
    assert state["blocks"] == 1


def test_budget_spent_passes():
    code, _, _ = sup.decide(run_cfg(), led([0.7] * 5), {}, NOW)
    assert code == 0


def test_plateau_detected():
    # best 0.85 early, then two candidates that don't beat it
    assert sup.plateaued(led([0.85, 0.849, 0.848])["candidates"], 2, 0.001)
    # improving run: no plateau
    assert not sup.plateaued(led([0.7, 0.8, 0.85])["candidates"], 2, 0.001)
    # too few candidates: no plateau
    assert not sup.plateaued(led([0.85, 0.84])["candidates"], 2, 0.001)


def test_plateau_injects_strategy_switch():
    code, msg, _ = sup.decide(run_cfg(), led([0.85, 0.849, 0.848]), {}, NOW)
    assert code == 2
    assert "CHANGE STRATEGY CLASS" in msg


def test_plateau_after_max_switches_exhausts():
    code, _, _ = sup.decide(run_cfg(), led([0.85, 0.849, 0.848], switches=2), {}, NOW)
    assert code == 0


def test_max_blocks_rail():
    code, _, _ = sup.decide(run_cfg(), led([0.7]), {"blocks": 10}, NOW)
    assert code == 0


def test_no_progress_rail():
    # first no-progress block passes through the rail (counter 1), second stops
    state = {"last_count": 1, "blocks": 1}
    code, _, state = sup.decide(run_cfg(), led([0.7]), state, NOW)
    assert code == 2 and state["no_progress_blocks"] == 1
    code, _, state = sup.decide(run_cfg(), led([0.7]), state, NOW)
    assert code == 0 and state["no_progress_blocks"] == 2


def test_progress_resets_rail():
    state = {"last_count": 1, "no_progress_blocks": 1, "blocks": 1}
    code, _, state = sup.decide(run_cfg(), led([0.7, 0.71]), state, NOW)
    assert code == 2 and state["no_progress_blocks"] == 0


def test_main_inert_without_config(tmp_path):
    """End-to-end: no run.json → exit 0, no output."""
    p = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"cwd": str(tmp_path), "session_id": "s"}),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and p.stderr == ""


def test_main_blocks_and_logs(tmp_path):
    """End-to-end: armed run with budget → exit 2, directive on stderr, log written."""
    supdir = tmp_path / ".supervisor"
    supdir.mkdir()
    (supdir / "run.json").write_text(json.dumps(run_cfg(ledger="ledger.json")))
    (supdir / "ledger.json").write_text(json.dumps(led([0.7])))
    p = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"cwd": str(tmp_path), "session_id": "s1"}),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2
    assert "SUPERVISOR" in p.stderr
    assert (supdir / "supervisor-log.jsonl").exists()
    assert json.loads((supdir / "state-s1.json").read_text())["blocks"] == 1


def test_main_kill_file(tmp_path):
    supdir = tmp_path / ".supervisor"
    supdir.mkdir()
    (supdir / "run.json").write_text(json.dumps(run_cfg(ledger="ledger.json")))
    (supdir / "ledger.json").write_text(json.dumps(led([0.7])))
    (supdir / "STOP").write_text("")
    p = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=json.dumps({"cwd": str(tmp_path), "session_id": "s2"}),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and p.stderr == ""
