"""Run every arm x kernel, verify checksums against native CPython, time it.

Arm contract (arms/<arm>/):
  build.sh               idempotent; builds artifacts into arms/<arm>/out/
  run.sh KERNEL N        runs one kernel; its LAST stdout line is the JSON the
                         kernel's __main__ prints: {"kernel","n","checksum","seconds"}
                         "seconds" is timed inside the program around run(n), so it
                         excludes runtime startup; bench.py also records wall time.
  artifact.txt           optional: one path per line (relative to the arm dir) whose
                         bytes count as the shipped size for each kernel, with
                         {kernel} substituted.
  SKIP                   optional: a file whose presence excludes the arm, with the reason.

Usage: python3 bench.py [--reps 5] [--arms a,b] [--kernels k1,k2]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNELS = ["nbody", "mandelbrot", "matmul", "dictwords"]
REL_TOL = 1e-9


def reference() -> dict:
    """Checksums from native CPython 3.11, computed fresh so they cannot drift."""
    ref = {}
    for k in KERNELS:
        out = subprocess.run([sys.executable if sys.version_info[:2] == (3, 11) else "python3.11",
                              str(ROOT / "kernels" / f"{k}.py")],
                             capture_output=True, text=True, check=True).stdout
        ref[k] = json.loads(out.strip().splitlines()[-1])
    return ref


def artifact_bytes(arm: Path, kernel: str):
    spec = arm / "artifact.txt"
    if not spec.exists():
        return None
    total = 0
    for line in spec.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = arm / line.format(kernel=kernel)
        if not p.exists():
            return None
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return total


def run_once(arm: Path, kernel: str, n: int, runtime: str):
    env = dict(os.environ, RUNTIME=runtime) if runtime else None
    t0 = time.perf_counter()
    p = subprocess.run(["bash", str(arm / "run.sh"), kernel, str(n)],
                       capture_output=True, text=True, timeout=900, cwd=arm, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return None, wall, (p.stderr or p.stdout)[-800:]
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    try:
        rec = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return None, wall, "unparseable stdout: " + p.stdout[-400:]
    return rec, wall, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default="")
    ap.add_argument("--kernels", default="")
    ap.add_argument("--runtime", default="", help="wasmtime|node; exported as RUNTIME to run.sh")
    ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    a = ap.parse_args()
    kernels = a.kernels.split(",") if a.kernels else KERNELS
    arms = sorted(d for d in (ROOT / "arms").iterdir() if (d / "run.sh").exists())
    if a.arms:
        want = a.arms.split(",")
        arms = [d for d in arms if d.name in want]
    ref = reference()
    rows = []
    for arm in arms:
        if (arm / "SKIP").exists():
            rows.append({"arm": arm.name, "skipped": (arm / "SKIP").read_text().strip()})
            print(f"{arm.name}: SKIP")
            continue
        for k in kernels:
            n = ref[k]["n"]
            row = {"arm": arm.name, "kernel": k, "n": n, "bytes": artifact_bytes(arm, k)}
            secs, walls, err = [], [], None
            for _ in range(a.reps):
                rec, wall, err = run_once(arm, k, n, a.runtime)
                if rec is None:
                    break
                exp = ref[k]["checksum"]
                if rec.get("n") != n or abs(rec["checksum"] - exp) > REL_TOL * max(1.0, abs(exp)):
                    err = f"checksum mismatch: got {rec.get('checksum')} n={rec.get('n')}, want {exp}"
                    break
                secs.append(rec["seconds"])
                walls.append(wall)
            if err:
                row["error"] = err
                print(f"{arm.name:24s} {k:11s} ERROR {err[:120]}")
            else:
                row.update(seconds=statistics.median(secs), wall=statistics.median(walls),
                           seconds_all=secs, verified=True)
                print(f"{arm.name:24s} {k:11s} {row['seconds']:8.3f}s  wall {row['wall']:8.3f}s")
            rows.append(row)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({"runtime": a.runtime or "wasmtime", "reference": ref, "rows": rows}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
