#!/usr/bin/env python3
"""Reduce a failing C file to a minimal repro.

    python3 tests/reduce.py BUG.c --mismatch            [--cc 'python3 -m hcc'] [-o OUT.c]
    python3 tests/reduce.py BUG.c --compile-fail 'KeyError'   # cc -S fails, stderr matches regex
    python3 tests/reduce.py BUG.c --predicate './my-check.sh' # exit 0 = still interesting

Modes (pick one):
  --mismatch          gcc -O0 build runs cleanly (also under ASan/UBSan, so the
                      reducer can't introduce UB) but the --cc build's exit code
                      or stdout differs (or it fails to compile/assemble/link/run).
  --compile-fail [RE] --cc fails to compile (-S) and stderr matches RE (default:
                      any failure); gcc must still accept the file.
  --predicate CMD     custom shell command run with the candidate path as $1.

Uses cvise if installed (unless --no-cvise), else a built-in line- then
token-level delta-debugging reducer.  Output defaults to BUG.reduced.c.
-I DIR adds include dirs (csmith failures get theirs from <seed>.json).
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS)
import run as runner  # noqa: E402

# Reference must compile without these (keeps reductions well-defined).  No -w: it disables them.
GCC_STRICT = ["-Werror=implicit-function-declaration", "-Werror=implicit-int",
              "-Werror=return-type", "-Werror=incompatible-pointer-types", "-Werror=int-conversion"]


# ------------------------------------------------------------ predicates

def build_cc(cc_argv, src, incs, work):
    """Build with the compiler under test like run.py does. Returns (exe or None, stderr)."""
    asm, obj, exe = (os.path.join(work, n) for n in ("cc.s", "cc.o", "cc.exe"))
    flags = ["-w"] + ["-I" + i for i in incs]
    rc, _, err, to = runner.run_cmd(cc_argv + flags + ["-S", "-o", asm, src],
                                    runner.COMPILE_TIMEOUT, env=runner.cc_env())
    if rc != 0 or to or not os.path.exists(asm):
        return None, runner.text(err) + ("\n<timeout>" if to else "")
    rc, _, err, _ = runner.run_cmd(["as", "--64", "-o", obj, asm], 60)
    if rc != 0:
        return None, runner.text(err)
    rc, _, err, _ = runner.run_cmd(["gcc", "-no-pie", "-o", exe, obj, "-lm"], 60)
    if rc != 0:
        return None, runner.text(err)
    return exe, ""


def check_mismatch(src, cc_argv, incs, sanitize):
    work = tempfile.mkdtemp(prefix="reduce-")
    try:
        inc = ["-I" + i for i in incs]
        ref = os.path.join(work, "gcc.exe")
        rc, _, _, to = runner.run_cmd(["gcc", "-O0"] + GCC_STRICT + inc + ["-o", ref, src, "-lm"], 60)
        if rc != 0 or to:
            return False
        rc_ref, out_ref, _, to = runner.run_cmd([ref], runner.RUN_TIMEOUT, cwd=work)
        if to or rc_ref < 0:
            return False
        if sanitize:
            san = os.path.join(work, "san.exe")
            rc, _, _, _ = runner.run_cmd(["gcc", "-O0", "-w", "-fsanitize=address,undefined",
                                          "-fno-sanitize-recover=all"] + inc + ["-o", san, src, "-lm"], 60)
            if rc != 0:
                return False
            rc, out, _, to = runner.run_cmd([san], runner.RUN_TIMEOUT * 3, cwd=work)
            if to or rc != rc_ref or out != out_ref:
                return False
        exe, _ = build_cc(cc_argv, src, incs, work)
        if exe is None:
            return True  # cc can't build a program gcc builds: still a bug
        rc, out, _, to = runner.run_cmd([exe], runner.RUN_TIMEOUT, cwd=work)
        return to or rc != rc_ref or out != out_ref
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check_compile_fail(src, cc_argv, incs, regex):
    inc = ["-I" + i for i in incs]
    rc, _, _, _ = runner.run_cmd(["gcc", "-fsyntax-only"] + GCC_STRICT + inc + [src], 60)
    if rc != 0:
        return False
    with tempfile.TemporaryDirectory(prefix="reduce-") as work:
        rc, _, err, to = runner.run_cmd(cc_argv + ["-w"] + inc + ["-S", "-o", os.path.join(work, "t.s"), src],
                                        runner.COMPILE_TIMEOUT, env=runner.cc_env())
    if rc == 0 and not to:
        return False
    return regex is None or re.search(regex, runner.text(err)) is not None


def check(args, path):
    if args.predicate:
        rc, _, _, _ = runner.run_cmd(["sh", "-c", args.predicate + ' "$1"', "pred", path], 300)
        return rc == 0
    cc_argv = shlex.split(args.cc)
    if args.mismatch:
        return check_mismatch(path, cc_argv, args.include, not args.no_sanitize)
    return check_compile_fail(path, cc_argv, args.include, args.compile_fail or None)


# -------------------------------------------------------------- reducers

class Reducer:
    """Delta debugging over a list of chunks (lines, then tokens)."""

    def __init__(self, args, out):
        self.args, self.out, self.tests = args, out, 0
        self.deadline = time.time() + args.max_time

    def interesting(self, text):
        self.tests += 1
        with open(self.out + ".tmp.c", "w") as f:
            f.write(text)
        return check(self.args, self.out + ".tmp.c")

    def ddmin(self, units):
        n = max(1, len(units) // 2)
        while n >= 1 and time.time() < self.deadline:
            i, progressed = 0, False
            while i < len(units) and time.time() < self.deadline:
                cand = units[:i] + units[i + n:]
                if cand and self.interesting("".join(cand)):
                    units = cand
                    progressed = True
                    with open(self.out, "w") as f:
                        f.write("".join(units))
                else:
                    i += n
            if not progressed:
                n //= 2
        return units

    def reduce(self, text):
        size = -1
        while len(text) != size and time.time() < self.deadline:
            size = len(text)
            text = "".join(self.ddmin(text.splitlines(keepends=True)))
            toks = re.findall(r"\s+|\w+|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|.", text)
            text = "".join(self.ddmin(toks))
            print("reduce: %d bytes after %d tests" % (len(text), self.tests), flush=True)
        if os.path.exists(self.out + ".tmp.c"):
            os.remove(self.out + ".tmp.c")
        return text


def run_cvise(args, src_text, out):
    work = tempfile.mkdtemp(prefix="cvise-")
    target = os.path.join(work, "repro.c")
    with open(target, "w") as f:
        f.write(src_text)
    script = os.path.join(work, "interesting.sh")
    me = [sys.executable, os.path.abspath(__file__), "--check"] + args.passthrough
    with open(script, "w") as f:
        f.write("#!/bin/sh\nexec %s repro.c\n" % shlex.join(me))
    os.chmod(script, 0o755)
    subprocess.run(["cvise", "--n", str(args.jobs), "--timeout", "300", "--tidy", script, "repro.c"], cwd=work)
    with open(target) as f:
        text = f.read()
    shutil.rmtree(work, ignore_errors=True)
    return text


def main():
    ap = argparse.ArgumentParser(description="reduce a failing C file (see module doc)")
    ap.add_argument("file")
    ap.add_argument("--cc", default="python3 -m hcc")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mismatch", action="store_true")
    mode.add_argument("--compile-fail", nargs="?", const="", metavar="REGEX")
    mode.add_argument("--predicate", metavar="CMD")
    ap.add_argument("-I", "--include", action="append", default=[])
    ap.add_argument("-o", "--output")
    ap.add_argument("--no-cvise", action="store_true", help="use the built-in reducer")
    ap.add_argument("--no-sanitize", action="store_true", help="skip ASan/UBSan check of reference")
    ap.add_argument("--preprocess", action="store_true",
                    help="flatten with gcc -E -P first (lets the reducer shrink headers; not for preprocessor bugs)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-time", type=int, default=1800, help="built-in reducer time budget (s)")
    ap.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    meta = os.path.splitext(args.file)[0] + ".json"  # csmith failure metadata
    if os.path.exists(meta) and not args.include:
        with open(meta) as f:
            inc = json.load(f).get("include")
        if inc:
            args.include = [inc]
    args.include = [os.path.abspath(i) for i in args.include]
    args.passthrough = ["--cc", args.cc] + sum((["-I", i] for i in args.include), [])
    if args.mismatch:
        args.passthrough.append("--mismatch")
    elif args.predicate:
        args.passthrough += ["--predicate", args.predicate]
    else:
        args.passthrough.append("--compile-fail=" + args.compile_fail)
    if args.no_sanitize:
        args.passthrough.append("--no-sanitize")

    if args.check:  # interestingness test entry point (used by cvise)
        sys.exit(0 if check(args, args.file) else 1)

    with open(args.file) as f:
        text = f.read()
    if args.preprocess:
        rc, out, err, _ = runner.run_cmd(["gcc", "-E", "-P"] + ["-I" + i for i in args.include] + [args.file], 60)
        if rc != 0:
            sys.exit("gcc -E failed: " + runner.text(err))
        text = runner.text(out)
    out = args.output or os.path.splitext(args.file)[0] + ".reduced.c"
    with open(out, "w") as f:
        f.write(text)
    if not check(args, out):
        sys.exit("reduce: the input is not interesting under this predicate; nothing to do")
    if shutil.which("cvise") and not args.no_cvise:
        text = run_cvise(args, text, out)
    else:
        text = Reducer(args, out).reduce(text)
    with open(out, "w") as f:
        f.write(text)
    print("reduce: wrote %s (%d lines, %d bytes)" % (out, text.count("\n"), len(text)))


if __name__ == "__main__":
    main()
