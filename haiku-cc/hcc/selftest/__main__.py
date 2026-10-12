"""hcc self tests: `python3 -m hcc.selftest [-k SUBSTR] [-j N] [-v]`.

Each hcc/selftest/*.c is compiled with hcc, linked, run. Header comments:
    // EXPECT-EXIT: 42          expected exit status (default 0)
    // EXPECT-STDOUT: text      one per expected stdout line (optional)
    // EXPECT-ERROR: [parse]    compilation must fail with this in stderr
    // EXPECT-FLAGS: -std=c23  extra hcc flags for this test
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def parse_expect(path: str) -> dict:
    exp = {"exit": 0, "stdout": None, "error": None, "flags": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("//"):
                if s:
                    break
                continue
            body = s[2:].strip()
            if body.startswith("EXPECT-EXIT:"):
                exp["exit"] = int(body.split(":", 1)[1])
            elif body.startswith("EXPECT-STDOUT:"):
                text = line.split("EXPECT-STDOUT:", 1)[1].rstrip("\n")
                text = text[1:] if text.startswith(" ") else text
                exp["stdout"] = (exp["stdout"] or []) + [text]
            elif body.startswith("EXPECT-ERROR:"):
                exp["error"] = body.split(":", 1)[1].strip()
            elif body.startswith("EXPECT-FLAGS:"):     # extra hcc command-line flags
                exp["flags"] = body.split(":", 1)[1].split()
    return exp


def run_one(path: str) -> tuple[str, bool, str]:
    name = os.path.basename(path)
    exp = parse_expect(path)
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "a.out")
        r = subprocess.run([sys.executable, "-m", "hcc", *exp["flags"], path, "-o", exe], cwd=ROOT,
                           capture_output=True, text=True, timeout=120)
        if exp["error"] is not None:
            ok = r.returncode != 0 and exp["error"] in r.stderr
            return name, ok, "" if ok else f"expected compile error {exp['error']!r}, got: {r.stderr.strip()[:300]}"
        if r.returncode != 0:
            return name, False, "compile failed: " + r.stderr.strip()[-600:]
        try:
            p = subprocess.run([exe], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return name, False, "timeout"
        if p.returncode != exp["exit"]:
            return name, False, f"exit {p.returncode}, expected {exp['exit']}; stdout={p.stdout[-200:]!r}"
        if exp["stdout"] is not None:
            want = "\n".join(exp["stdout"]) + "\n"
            if p.stdout != want:
                return name, False, f"stdout {p.stdout!r}, expected {want!r}"
    return name, True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 2)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(HERE, "*.c")) if a.k in os.path.basename(f))
    failed = 0
    with cf.ThreadPoolExecutor(a.j) as ex:
        for name, ok, msg in ex.map(run_one, files):
            if not ok:
                failed += 1
                print(f"FAIL {name}: {msg}")
            elif a.v:
                print(f"PASS {name}")
    print(f"{len(files) - failed}/{len(files)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
