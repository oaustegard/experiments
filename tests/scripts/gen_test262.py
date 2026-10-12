#!/usr/bin/env python3
"""Select tc39/test262 tests for the quickjs suite and commit them under
tests/suites/quickjs/test262/ (+ the harness in test262-harness/).

    python3 tests/scripts/gen_test262.py [--src tests/tools/test262-src] [--per-dir 4] [--cap 4000]

Selection: every directory under test/language, test/built-ins and test/annexB contributes up
to --per-dir tests, evenly spaced through its sorted file list (so the whole spec surface is
covered, not just the first files).  Module tests are skipped.  Each test is then assembled
(tests/drivers/quickjs.py:assemble_test262) and run under the gcc -O0 build of QuickJS; it is kept
only when that outcome is the one test262 expects (exit 0 for positive tests, Test262:AsyncTestComplete
for async, a failure naming the expected error type for negative tests).  Tests that need host
features plain qjs lacks ($262 realms, agents, ...) therefore drop out.

The test262 clone is fetched with:  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/tc39/test262 tests/tools/test262-src
(commit recorded in test262/COMMIT).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.join(TESTS, "drivers"))
import run as R  # noqa: E402
import quickjs as Q  # noqa: E402

ROOTS = ["language", "built-ins", "annexB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(TESTS, "tools", "test262-src"))
    ap.add_argument("--per-dir", type=int, default=4)
    ap.add_argument("--cap", type=int, default=4000)
    args = ap.parse_args()
    suite = R.Suite("quickjs")
    out_root = os.path.join(suite.dir, "test262")
    hdir = Q.harness_dir(suite)
    shutil.rmtree(out_root, ignore_errors=True)
    shutil.rmtree(hdir, ignore_errors=True)
    os.makedirs(hdir)
    for f in os.listdir(os.path.join(args.src, "harness")):
        p = os.path.join(args.src, "harness", f)
        if f.endswith(".js") and os.path.isfile(p):
            shutil.copy(p, hdir)
    ref = Q.ref_qjs(R, suite)

    cands = []
    for root in ROOTS:
        base = os.path.join(args.src, "test", root)
        for dp, dn, fn in os.walk(base):
            dn.sort()
            files = sorted(f for f in fn if f.endswith(".js") and "_FIXTURE" not in f)
            if not files:
                continue
            k = min(args.per_dir, len(files))
            idx = sorted({int(i * len(files) / k) for i in range(k)})
            cands += [os.path.join(dp, files[i]) for i in idx]
    print("%d candidates" % len(cands), flush=True)

    kept = 0
    work = tempfile.mkdtemp(prefix="t262-")
    sel = []
    for n, path in enumerate(cands):
        rel = os.path.relpath(path, os.path.join(args.src, "test"))
        try:
            script, meta = Q.assemble_test262(suite, path)
        except (OSError, UnicodeDecodeError):
            continue
        if "module" in meta["flags"]:
            continue
        with open(os.path.join(work, "t.js"), "w", encoding="utf-8") as f:
            f.write(script)
        r = subprocess.run([ref, "t.js"], cwd=work, capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, TZ="UTC", LC_ALL="C"), errors="replace")
        if meta["negative"]:
            ok = r.returncode != 0 and meta["negative"].get("type", "") in r.stderr
        elif "async" in meta["flags"]:
            ok = r.returncode == 0 and "Test262:AsyncTestComplete" in r.stdout
        else:
            ok = r.returncode == 0
        if ok:
            sel.append((rel, path))
        if n % 500 == 0:
            print("%d/%d checked, %d kept" % (n, len(cands), len(sel)), flush=True)
    if len(sel) > args.cap:
        step = len(sel) / args.cap
        sel = [sel[int(i * step)] for i in range(args.cap)]
    for rel, path in sel:
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(path, dst)
    commit = subprocess.run(["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out_root, "COMMIT"), "w") as f:
        f.write("tc39/test262 %s\n" % commit)
    shutil.rmtree(work, ignore_errors=True)
    print("kept %d tests in %s (%.1f MB)" % (len(sel), out_root, sum(os.path.getsize(p) for _, p in sel) / 1e6))


if __name__ == "__main__":
    main()
