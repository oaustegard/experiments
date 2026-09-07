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
POOL_N = int(sys.argv[1]) if len(sys.argv) > 1 else 9
KEEP_N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/tasks-pilot.json"
SEED = 20260906


def run_one(lang, task, variant):
    dst = ROOT / "work" / "_certify" / variant / lang / task
    shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bench.ex_dir(lang, task), dst)
    if variant == "gold":
        sols = bench.solution_files(lang, task)
        exs = bench.cfg(lang, task)["files"].get("example") or []
        # cpp declares two solution files (.cpp and .h) and two examples; java
        # nests its reference under .meta/src/reference/java. Match positionally
        # when the config lists them, and fall back to a glob when it does not.
        if len(exs) == len(sols):
            pairs = list(zip(exs, sols))
        else:
            cand = sorted(c for c in (dst / ".meta").rglob("example*") if c.is_file())
            if not cand:
                return None, "no example file"
            pairs = [(str(cand[0].relative_to(dst)), sols[0])]
        for src_rel, dst_rel in pairs:
            src_p = dst / src_rel
            if not src_p.exists():
                return None, f"example missing: {src_rel}"
            (dst / dst_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst / dst_rel)
    spec = bench.LANGS[lang]
    env = dict(os.environ, GOFLAGS="-mod=mod", GOPATH="/tmp/gopath",
               CARGO_TARGET_DIR=str(dst / "_target"))
    if lang == "java":
        env["GRADLE_USER_HOME"] = os.environ.get("GRADLE_USER_HOME", "/root/.gradle")
    try:
        r = subprocess.run(spec["cmd"], cwd=dst, capture_output=True, text=True,
                           timeout=spec["timeout"], env=env)
        return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        for junk in ("_target", "node_modules", "build", ".gradle"):
            shutil.rmtree(dst / junk, ignore_errors=True)


rng = random.Random(SEED)
report, chosen = {}, []
for lang in os.environ.get("LANGS", "python,go,rust").split(","):
    pool = sorted(p.name for p in (PG / lang / "exercises" / "practice").iterdir() if p.is_dir())
    # shuffle the WHOLE pool once and walk it in order, so the admitted set for
    # KEEP_N=4 is a prefix of the set for KEEP_N=10. rng.sample(pool, k) is not
    # stable across k and silently re-draws the task set when the size changes.
    rng.shuffle(pool)
    cands = pool[:min(POOL_N, len(pool))]
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

(ROOT / OUT.replace("tasks-", "certify-")).write_text(json.dumps(report, indent=1))
(ROOT / OUT).write_text(json.dumps(chosen, indent=1))
print(f"\nadmitted {len(chosen)}: " + ", ".join(f"{c['lang']}/{c['task']}" for c in chosen))
