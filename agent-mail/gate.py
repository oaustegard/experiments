"""Handoff obligations and the SHIP gate for an agent-mail board.

The coordinator seeds the obligations it created when it split the spec
(`seed FROM TO`). A holder discharges one with `owe`, which sends the rule and
attaches one test case. The owner answers with `ack` after implementing it.
`ship` sends SHIP only when every obligation is relayed, acknowledged, and its
test passes against the team's current tree; otherwise it tells each
responsible agent what is still open.

    obligations.json   {id: {from, to, rule, test, msg, acked}}
    fixture.json       the catalog every test runs against (written by the coordinator)
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _locked(b: Path):
    with open(b / ".obligations.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        f = b / "obligations.json"
        obs = json.loads(f.read_text()) if f.exists() else {}
        yield obs
        f.write_text(json.dumps(obs, indent=1))


def seed(b: Path, holder: str, owner: str) -> str:
    with _locked(b) as obs:
        oid = f"o{len(obs) + 1}"
        obs[oid] = {"from": holder, "to": owner, "rule": None, "test": None, "msg": None, "acked": False}
    return oid


def owe(b: Path, send, holder: str, owner: str, rule: str, test: dict) -> str:
    """Record (or discharge a seeded) obligation and send the rule; returns the obligation id."""
    with _locked(b) as obs:
        oid = next((k for k, o in obs.items() if o["from"] == holder and o["to"] == owner and o["rule"] is None), None)
        oid = oid or f"o{len(obs) + 1}"
        msg = send(holder, owner, f"RULE {oid} for your file: {rule}  |  test: {json.dumps(test)}  |  "
                                  f"implement it, then run: mail ack {owner} {oid}")
        obs[oid] = {"from": holder, "to": owner, "rule": rule, "test": test, "msg": msg, "acked": False}
    return oid


def ack(b: Path, owner: str, oid: str) -> str | None:
    with _locked(b) as obs:
        o = obs.get(oid)
        if o is None or o["to"] != owner:
            return f"no obligation {oid} addressed to {owner}"
        o["acked"] = True
    return None


def run_test(tree: Path, catalog: dict, test: dict) -> str | None:
    """None when the owner's tree produces the expected fragment; else the reason."""
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); from orders.report import process; "
            "t=json.loads(sys.argv[2]); print(process(t['csv'], json.loads(sys.argv[3]), t.get('stock', {})))")
    try:
        r = subprocess.run([sys.executable, "-c", code, str(tree), json.dumps(test), json.dumps(catalog)],
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "test timed out"
    if r.returncode != 0:
        return "process() raised: " + (r.stderr.strip().splitlines() or ["?"])[-1]
    out = r.stdout
    missing = [s for s in test.get("expect", []) if s not in out]
    present = [s for s in test.get("absent", []) if s in out]
    if missing or present:
        return f"expected {missing} absent {present}; got: {out[:300]!r}"
    return None


def ship(b: Path, send, name: str) -> list[str]:
    """Send SHIP and return [] when every obligation holds; else notify owners and return the problems."""
    tree = b.parent
    fx = b / "fixture.json"
    catalog = json.loads(fx.read_text()) if fx.exists() else {}
    with _locked(b) as obs:
        snapshot = json.loads(json.dumps(obs))
    problems = []
    for oid, o in snapshot.items():
        if o["rule"] is None:
            problems.append((o["from"], f"{oid}: you hold a rule for {o['to']}'s file and have not sent it; "
                                        f"use: mail owe {o['from']} {o['to']} \"<rule>\" '<test json>'"))
        elif not o["acked"]:
            problems.append((o["to"], f"{oid}: rule from {o['from']} not acknowledged: {o['rule']}"))
        else:
            why = run_test(tree, catalog, o["test"])
            if why:
                problems.append((o["to"], f"{oid}: test for {o['from']}'s rule fails: {why}"))
    for who, text in problems:
        send(name, who, "SHIP blocked: " + text)
    if not problems:
        send(name, "all", "SHIP")
    return [f"{who}: {text}" for who, text in problems]


def status(b: Path) -> list[str]:
    with _locked(b) as obs:
        return [f"{k} {o['from']}->{o['to']} relayed={o['rule'] is not None} acked={o['acked']} "
                f"rule={o['rule']!r}" for k, o in obs.items()]
