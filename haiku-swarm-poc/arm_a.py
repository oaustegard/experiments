"""Arm A: stock OpenEvolve 0.4.1 on Sonnet 5.5 through an OpenAI-compatible endpoint.

The config is OpenEvolve's examples/circle_packing/config_phase_1.yaml with
only the LLM section and the stopping rule changed: one model
(claude-sonnet-5-5, weight 1.0) at the gateway's endpoint, max_iterations
lifted so that the dollar cap and the 60-minute wall clock are what stop the
run, and random_seed set per seed. OpenEvolve runs inside the jail; the
driver kills its process group when the ledger reaches the cap or the clock
runs out.

Usage: python3 arm_a.py RUN_ID --seed S --cap 3 [--wall 3600]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import yaml

import problems
from common import JAIL, give_to_nobody, jail_cmd, run_dir, score_of, setup_jail
from gateway import Gateway, serve_openai

MODEL = "claude-sonnet-5-5"
EFFORT = "high"   # Sonnet 5.5's API default; OpenEvolve sends no effort of its own
PORT = 18080


def write_config(d: Path, seed: int, spec: dict):
    cfg = yaml.safe_load((JAIL / spec["jail_dir"] / spec["config"]).read_text())
    llm = cfg["llm"]
    for k in ("primary_model", "primary_model_weight", "secondary_model", "secondary_model_weight", "models"):
        llm.pop(k, None)
    llm.update(primary_model=MODEL, primary_model_weight=1.0,
               api_base=f"http://127.0.0.1:{PORT}/v1", api_key="unused-local-gateway")
    if spec["name"] != "circle":
        # The AlphaEvolve examples were configured for gpt-4 (max_tokens 4000, timeout
        # 120 s). A reasoning model needs room for its thinking, or every long answer
        # fails and is billed anyway; circle packing's config already had 8192 / 600.
        llm.update(max_tokens=16000, timeout=600)
    cfg["max_iterations"] = 1_000_000
    cfg["random_seed"] = seed
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def kill_group(proc: subprocess.Popen):
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def build_events(d: Path, t0: float, problem: str = "circle") -> list[dict]:
    """One event per program: its last evaluation stage, at the spend reached by then."""
    rows = [json.loads(l) for l in (d / "evals.jsonl").read_text().splitlines()] if (d / "evals.jsonl").exists() else []
    rows.sort(key=lambda r: r["t"])
    ledger = [json.loads(l) for l in (d / "ledger.jsonl").read_text().splitlines()] if (d / "ledger.jsonl").exists() else []
    ledger = [r for r in ledger if r.get("role") != "calibrate"]
    ledger.sort(key=lambda r: r["t_end"])
    events = []
    for i, r in enumerate(rows):
        if r["fn"] == "stage1" and any(x["sha"] == r["sha"] and x["fn"] == "stage2" for x in rows[i + 1:]):
            continue
        spent = sum(l["cost"] for l in ledger if l["t_end"] <= r["t"])
        spent_api = sum(l["cost_api"] for l in ledger if l["t_end"] <= r["t"])
        m = r["metrics"]
        events.append({"kind": "eval", "sha": r["sha"], "score": score_of(m, problem), "te": r["t"] - t0,
                       "spent": spent, "spent_api": spent_api, "stage": r["fn"],
                       "error": str(m.get("error", ""))[:200] or None})
    return events


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--wall", type=float, default=3600)
    ap.add_argument("--problem", default="circle", choices=list(problems.PROBLEMS))
    ap.add_argument("--effort", default=EFFORT)
    a = ap.parse_args(argv)
    setup_jail()
    d = run_dir(a.run_id)
    if (d / "DONE").exists():
        print("already done:", a.run_id)
        return json.loads((d / "DONE").read_text())
    if any(d.iterdir()):
        # OpenEvolve cannot resume against our clock and ledger; restart clean.
        shutil.move(str(d), str(d.parent / f"{a.run_id}.partial-{int(time.time())}"))
        d = run_dir(a.run_id)

    gw = Gateway(d, a.cap, max_parallel=8)
    calib = JAIL / "runs" / "_calib" / "overhead.json"
    if calib.exists():
        gw.overhead = json.loads(calib.read_text())
    spec = problems.get(a.problem)
    write_config(d, a.seed, spec)
    srv = serve_openai(gw, str(d / "llm.sock"), effort=a.effort)
    give_to_nobody(d)
    rel = d.relative_to(JAIL)
    log = open(d / "driver.log", "w")
    t0 = time.time()
    (d / "t0").write_text(str(t0))
    proc = subprocess.Popen(jail_cmd("sh", "jailside/run_oe.sh", str(rel),
                                     env={"HSP_EVAL_LOG": f"{JAIL}/{rel}/evals.jsonl",
                                          "HSP_PROBLEM_DIR": spec["jail_dir"]}),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stopped_by = "exit"
    while True:
        time.sleep(5)
        if proc.poll() is not None:
            break
        if gw.spent >= a.cap:
            stopped_by = "cap"
            break
        if time.time() - t0 >= a.wall:
            stopped_by = "wall"
            break
    elapsed = time.time() - t0
    kill_group(proc)
    srv.shutdown()
    events = build_events(d, t0, a.problem)
    with (d / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    ev = [e for e in events if e["te"] <= a.wall]
    best = max(ev, key=lambda e: e["score"]) if ev else None
    summary = {"run": a.run_id, "arm": "A", "problem": a.problem, "seed": a.seed, "model": MODEL, "effort": a.effort,
               "cap": a.cap, "spent": gw.spent, "spent_api": gw.spent_api, "calls": gw.calls,
               "elapsed": elapsed, "programs": len({e["sha"] for e in ev}),
               "best_score": best["score"] if best else None, "stopped_by": stopped_by,
               "oe_exit": proc.returncode}
    (d / "DONE").write_text(json.dumps(summary, indent=1))
    print("done ->", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
