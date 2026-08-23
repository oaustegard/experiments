"""Driver for the SPECTER2 quantization search — runs candidates, appends to
ledger.json, applies the same plateau rule the (unwired, in this session)
supervisor_stop.py hook would have: window=4, min_delta=0.001,
max_strategy_switches=3. Not a supervisor loop; a manual stand-in for one,
run directly since this session has no Stop hook wired to the hub's script.
"""
import json
import subprocess
import sys
import time

LEDGER = "/home/user/experiments/avo-supervisor-specter2/ledger.json"
FITNESS = ["python3", "/home/user/claude-workspace/.supervisor/fitness_specter2.py"]

WINDOW = 4
MIN_DELTA = 0.001


def load():
    with open(LEDGER) as f:
        return json.load(f)


def save(ledger):
    with open(LEDGER, "w") as f:
        json.dump(ledger, f, indent=2)


def plateaued(candidates):
    if len(candidates) <= WINDOW:
        return False
    scores = [c["score"] for c in candidates]
    best_before = max(scores[:-WINDOW])
    return all(s < best_before + MIN_DELTA for s in scores[-WINDOW:])


def run_one(args, desc, strategy_class):
    t0 = time.time()
    proc = subprocess.run(FITNESS + args, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        return None, proc.stderr
    score = float(proc.stdout.strip())
    ledger = load()
    entry = {
        "id": len(ledger["candidates"]) + 1,
        "ts": time.time(),
        "score": score,
        "desc": desc,
        "strategy_class": strategy_class,
        "wall_s": round(wall, 2),
        "args": args,
    }
    ledger["candidates"].append(entry)
    save(ledger)
    print(f"[{entry['id']:2d}] {strategy_class:20s} {desc:45s} score={score:.4f} wall={wall:.2f}s")
    return entry, None


if __name__ == "__main__":
    print("driver ready")
