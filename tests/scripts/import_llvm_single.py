#!/usr/bin/env python3
"""(Re)build tests/suites/llvm-single from llvm/llvm-test-suite SingleSource.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/llvm/llvm-test-suite.git tests/tools/src/llvm-test-suite
    (cd tests/tools/src/llvm-test-suite && git sparse-checkout set SingleSource)
    python3 tests/scripts/import_llvm_single.py [tests/tools/src/llvm-test-suite]

Covers SingleSource/UnitTests (subdirectories that hold portable C: not the per-architecture,
intrinsics and vectoriser directories), Regression/C (top level, SSE,
gcc-c-torture/execute/ieee; the rest of gcc-c-torture duplicates the gcc-torture suite and
builtins/ needs paired -lib.c files) and the Benchmarks that are plain C and quick
(BenchmarkGame, CoyoteBench, Dhrystone, Linpack, McGill, Misc, Shootout, Stanford).

Each program is built with gcc -O0 -w and run twice (stdin /dev/null, 10 s limit).  The
expectation is the gcc binary's stdout (<t>.c.expected, only if both runs agree) and exit code
(<t>.c.exitcode if non-zero): hcc must reproduce them.  The upstream `reference_output` is
compared with gcc's output (minus its trailing "exit N" line) and the verdict kept in
manifest.json, but gcc is the truth.  Skipped (skip.txt, with reasons): gcc cannot build it,
a gcc run takes more than 1.5 s (hcc's code is not optimised, and the runner's limit is 10 s),
it is killed by a signal, or it needs other files (arguments, stdin, extra sources).
"""
import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
DEST = os.path.join(TESTS, "suites", "llvm-single")
MAX_SECS = 1.5

ROOTS = [
    ("UnitTests", True),
    ("UnitTests/Vector", False),  # GCC vector extensions; the intrinsics subdirectories are left out
    ("Regression/C", False),
    ("Regression/C/SSE", False),
    ("Regression/C/gcc-c-torture/execute/ieee", False),
] + [("Benchmarks/" + d, False) for d in
     ("BenchmarkGame", "CoyoteBench", "Dhrystone", "Linpack", "McGill", "Misc", "Shootout", "Stanford")]


def collect(src_root):
    base = os.path.join(src_root, "SingleSource")
    files = []
    for rel, recursive in ROOTS:
        d = os.path.join(base, rel)
        if recursive:
            for dp, dn, fn in os.walk(d):
                dn[:] = sorted(x for x in dn if x not in ("C++11", "ObjC", "ObjC++", "EH", "AArch64", "Mips", "LoongArch", "PowerPC", "RISCV",
                                                 "SystemZ", "Hexagon", "ARM", "X86", "Vector", "Vectorizer", "Matrix"))
                for f in sorted(fn):
                    if f.endswith((".c", ".h")):
                        files.append(os.path.relpath(os.path.join(dp, f), base))
        else:
            for f in sorted(os.listdir(d)):
                if os.path.isfile(os.path.join(d, f)) and f.endswith((".c", ".h")):
                    files.append(os.path.join(rel, f))
    return base, files


def run(exe, timeout=10):
    t0 = time.time()
    try:
        p = subprocess.run([exe], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stdout, time.time() - t0
    except subprocess.TimeoutExpired:
        return None, b"", time.time() - t0


def ref_verdict(src_c, out):
    ref = src_c[:-2] + ".reference_output"
    if not os.path.exists(ref):
        return "none"
    want = open(ref, "rb").read()
    want = re.sub(rb"exit \d+\n?$", b"", want)
    return "match" if want == out else "differs"


def one(base, rel):
    """Returns (rel, skip_reason or None, info dict)."""
    src = os.path.join(base, rel)
    d = tempfile.mkdtemp(prefix="llvm-single-")
    try:
        exe = os.path.join(d, "t")
        r = subprocess.run(["gcc", "-O0", "-w", "-no-pie", "-o", exe, src, "-lm"],
                           capture_output=True, text=True, cwd=d)
        if r.returncode:
            err = [l for l in r.stderr.splitlines() if "error" in l] or r.stderr.splitlines() or ["?"]
            return rel, "gcc -O0 cannot build it (%s)" % re.sub(r"^.*?: ", "", err[0])[:80], {}
        rc1, out1, t1 = run(exe)
        if rc1 is None or t1 > MAX_SECS:
            return rel, "gcc -O0 run takes more than %.1fs (%s)" % (MAX_SECS, "timeout" if rc1 is None else "%.1fs" % t1), {}
        if rc1 < 0:
            return rel, "gcc -O0 binary dies of signal %d" % -rc1, {}
        rc2, out2, t2 = run(exe)
        det = rc1 == rc2 and out1 == out2
        if len(out1) > 200000:
            return rel, "output larger than 200 KB", {}
        info = {"rc": rc1, "det": det, "ref": ref_verdict(src, out1), "secs": round(max(t1, t2), 2)}
        return rel, None, dict(info, out=out1 if det else None)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    src_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TESTS, "tools", "src", "llvm-test-suite")
    base, files = collect(src_root)
    if os.path.isdir(DEST):
        for dp, _, fn in os.walk(DEST):
            for f in fn:
                if f.endswith((".c", ".h", ".expected", ".exitcode", ".flags")) or f in ("skip.txt", "manifest.json"):
                    os.remove(os.path.join(dp, f))
    skips, manifest, ntests = {}, {}, 0
    sources = [f for f in files if f.endswith(".c")]
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        results = list(ex.map(lambda f: one(base, f), sources))
    for f in files:  # headers next to the tests
        if f.endswith(".h"):
            os.makedirs(os.path.dirname(os.path.join(DEST, f)), exist_ok=True)
            shutil.copy(os.path.join(base, f), os.path.join(DEST, f))
    for rel, reason, info in results:
        ntests += 1
        dst = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(base, rel), dst)
        if reason:
            skips[rel] = reason
            continue
        if info["det"]:
            with open(dst + ".expected", "wb") as fh:
                fh.write(info["out"])
        if info["rc"]:
            with open(dst + ".exitcode", "w") as fh:
                fh.write("%d\n" % info["rc"])
        manifest[rel] = {k: v for k, v in info.items() if k != "out"}
    with open(os.path.join(DEST, "skip.txt"), "w") as fh:
        fh.write("# Generated by tests/scripts/import_llvm_single.py: <test>  # reason\n")
        for f in sorted(skips):
            fh.write("%s  # %s\n" % (f, skips[f]))
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=0, sort_keys=True)
    cfg = {"description": "llvm/llvm-test-suite SingleSource: UnitTests, Regression/C (+ieee), quick plain-C Benchmarks. "
                          "Expected stdout/exit code come from a gcc -O0 build (compared with the upstream "
                          "reference_output in manifest.json); output that differs between two gcc runs is not compared. "
                          "Regenerate with tests/scripts/import_llvm_single.py.",
           "glob": ["**/*.c"], "cflags": ["-w"]}
    with open(os.path.join(DEST, "suite.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    rev = subprocess.run(["git", "-C", src_root, "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(DEST, "SOURCE"), "w") as fh:
        fh.write("https://github.com/llvm/llvm-test-suite %s SingleSource\n" % rev)
    print("imported %d programs, %d skipped, %d active (%d with exact stdout)" % (
        ntests, len(skips), ntests - len(skips), sum(1 for m in manifest.values() if m["det"])))


if __name__ == "__main__":
    main()
