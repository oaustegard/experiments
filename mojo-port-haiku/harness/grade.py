"""Grade one agent's port on the held-out split.

Copies the agent's workdir (package + build.sh + .mojo sources, no build
outputs) to a fresh directory, rebuilds it with the agent's build.sh, then
runs the target's test-split oracle there, so a stale .so or an edited
oracle in the workdir cannot make a port look correct. For difflib it also
runs CPython's test_difflib inside claude-workspace's jail.

    python3 harness/grade.py --target snowball --workdir runs/r1/snowball [--bench]

Writes <workdir>/../grade-<target>.json and prints it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAIL = Path("/home/user/claude-workspace/scripts/jail.sh")
PKG = {"snowball": "msnowball", "difflib": "mdifflib", "yake": "myake"}
ORACLE = {
    "snowball": ROOT / "oracle/snowball/check.py",
    "difflib": ROOT / "oracle/difflib/fuzz.py",
    "yake": ROOT / "oracle/yake/check.py",
}
IGNORE = shutil.ignore_patterns("*.so", "__pycache__", "__mojocache__", "build", ".pixi", "*.o")


def run(cmd, cwd=None, timeout=1800) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


def last_json(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {"unparsed_output": text[-2000:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(PKG))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    work = Path(args.workdir).resolve()
    stage = Path("/tmp/jail") / f"mojo-port-grade-{args.target}-{work.parent.name}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(work, stage, ignore=IGNORE)
    out: dict = {"target": args.target, "workdir": str(work), "graded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    build = stage / "build.sh"
    if not build.exists():
        out.update(build_ok=False, build_log="no build.sh", all_pass=False)
    else:
        t = time.perf_counter()
        rc, log = run(["sh", "build.sh"], cwd=stage)
        out.update(build_ok=rc == 0, build_s=round(time.perf_counter() - t, 1), build_log=log[-1500:])

    if out.get("build_ok"):
        rc, log = run([sys.executable, str(ORACLE[args.target]), "--pkg", str(stage), "--split", "test"])
        out["oracle"] = last_json(log)
        ok = bool(out["oracle"].get("all_pass"))
        if args.target == "difflib":
            shutil.copytree(ROOT / "oracle/difflib", stage / "_upstream", dirs_exist_ok=True)
            run(["chown", "-R", "nobody:nogroup", str(stage)])
            rc, log = run(["sh", str(JAIL), "-C", str(stage), "--",
                           "/usr/bin/python3", "_upstream/run_upstream.py", "--pkg", "."])
            out["upstream"] = last_json(log)
            ok = ok and bool(out["upstream"].get("all_pass"))
        out["all_pass"] = ok
        if args.bench and ok:
            rc, log = run([sys.executable, str(ROOT / "bench/bench.py"), "--target", args.target,
                           "--pkg", str(stage), "--reps", "5"], timeout=3600)
            out["bench"] = last_json(log)

    dest = work.parent / f"grade-{args.target}.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out.get("all_pass") else 1


if __name__ == "__main__":
    sys.exit(main())
