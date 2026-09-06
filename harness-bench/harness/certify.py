#!/usr/bin/env python3
"""Admit a task only if the grader can go BOTH ways on it:
reference solution -> PASS, untouched stub -> FAIL.

Rejects two real classes seen on the first pilot draw:
  - refactoring exercises whose stub already passes (go/markdown)  -> no signal
  - reference solutions that need crates the exercise Cargo.toml lacks
    (rust/poker wants `counter`, rust/pig-latin wants `regex`)     -> not certifiable here
"""
import json, os, random, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import bench

ROOT, PG = bench.ROOT, bench.PG
POOL_N, KEEP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 9, 4
SEED = 20260906


def run_one(lang, task, variant):
    dst = ROOT / "work" / "_certify" / variant / lang / task
    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bench.ex_dir(lang, task), dst)
    if variant == "gold":
        sol = bench.solution_files(lang, task)[0]
        cand = [c for c in (dst / ".meta").rglob("example*") if c.is_file()]
        if not cand:
            return None, "no example file"
        shutil.copy2(cand[0], dst / sol)
    spec = bench.LANGS[lang]
    env = dict(os.environ, GOFLAGS="-mod=mod", GOPATH="/tmp/gopath",
               CARGO_TARGET_DIR=str(dst / "_target"))
    try:
        r = subprocess.run(spec["cmd"], cwd=dst, capture_output=True, text=True,
                           timeout=spec["timeout"], env=env)
        return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        shutil.rmtree(dst / "_target", ignore_errors=True)


rng = random.Random(SEED)
report, chosen = {}, []
for lang in ("python", "go", "rust"):
    pool = sorted(p.name for p in (PG / lang / "exercises" / "practice").iterdir() if p.is_dir())
    cands = rng.sample(pool, min(POOL_N, len(pool)))
    kept = []
    for task in cands:
        if len(kept) >= KEEP_N:
            report[f"{lang}/{task}"] = dict(verdict="not-needed")
            continue
        gold, gout = run_one(lang, task, "gold")
        stub, sout = run_one(lang, task, "stub")
        ok = (gold is True) and (stub is False)
        why = "admitted" if ok else ("gold-fails" if gold is not True else "stub-passes")
        report[f"{lang}/{task}"] = dict(verdict=why, gold=gold, stub=stub,
                                        gold_tail=None if ok else (gout or "")[-600:])
        print(f"{why:12} {lang}/{task}", flush=True)
        if ok:
            kept.append(task)
    chosen += [dict(lang=lang, task=t) for t in kept]

(ROOT / "results/certify.json").write_text(json.dumps(report, indent=1))
(ROOT / "results/tasks-pilot.json").write_text(json.dumps(chosen, indent=1))
print(f"\nadmitted {len(chosen)}: " + ", ".join(f"{c['lang']}/{c['task']}" for c in chosen))
