#!/usr/bin/env python3
"""Differential testing of --cc against gcc -O0 using csmith programs.

    python3 tests/csmith/diff.py --cc 'python3 -m hcc' --seeds 1:100 [--jobs 4] [--json OUT]

For each seed: generate tests/csmith/work/<seed>.c with csmith, build it with
gcc -O0 -w (reference) and with --cc (via -S, then as + gcc link, exactly like
tests/run.py), run both, and compare the printed "checksum = X" line.

One line per seed:  OK | MISMATCH | CRASH | COMPILE_FAIL | ASM_FAIL | LINK_FAIL |
                    RUN_FAIL | TIMEOUT | SKIP   <seed> <reason>
CRASH = the compiler crashed (traceback/signal); RUN_FAIL = hcc's binary
crashed; SKIP = csmith or the gcc reference itself failed/timed out.
Every non-OK, non-SKIP seed is saved to tests/csmith/failures/<seed>.c plus
<seed>.json metadata {seed, kind, reason, csmith_args, gcc_output, cc_output,
cc_stderr, cc}.  Reduce one with:
    python3 tests/reduce.py tests/csmith/failures/<seed>.c --mismatch --cc '...'
Build csmith first with tests/csmith/build.sh.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import run as runner  # noqa: E402  (tests/run.py: run_cmd, cc_env, stage_from_stderr)

CSMITH = os.path.join(TESTS, "tools", "csmith", "bin", "csmith")
WORK = os.path.join(HERE, "work")
FAILURES = os.path.join(HERE, "failures")

# Keep programs C11, small, and quick to compile with a Python compiler.
CSMITH_ARGS = ["--no-packed-struct", "--max-funcs", "4", "--max-block-depth", "3",
               "--max-expr-complexity", "6", "--max-struct-fields", "6",
               "--max-array-dim", "2", "--max-array-len-per-dim", "8"]
GEN_TIMEOUT = 30
RUN_TIMEOUT = 10


def csmith_include():
    hits = glob.glob(os.path.join(TESTS, "tools", "csmith", "include", "**", "csmith.h"),
                     recursive=True)
    if not hits:
        sys.exit("csmith not built: run tests/csmith/build.sh")
    return os.path.dirname(hits[0])


def checksum_line(out):
    for line in runner.text(out).splitlines():
        if line.startswith("checksum"):
            return line.strip()
    return None


def build_with(cc_argv, src, base, inc, split):
    """Compile+link src. Returns (kind, reason, stderr) where kind is None on success."""
    cflags = ["-w", "-I" + inc]
    if not split:  # reference compiler: one shot
        rc, _, err, to = runner.run_cmd(cc_argv + cflags + ["-O0", "-o", base, src, "-lm"], 120)
        return (None if rc == 0 and not to else "SKIP"), "gcc failed", runner.text(err)
    asm, obj = base + ".s", base + ".o"
    rc, _, err, to = runner.run_cmd(cc_argv + cflags + ["-S", "-o", asm, src],
                                    runner.COMPILE_TIMEOUT, env=runner.cc_env())
    err_s = runner.text(err)
    if to:
        return "TIMEOUT", "compile >%ds" % runner.COMPILE_TIMEOUT, err_s
    if rc != 0 or not os.path.exists(asm):
        crashed = rc < 0 or "Traceback (most recent call last)" in err_s
        stage = runner.stage_from_stderr(err_s) or "?"
        reason = "%s: %s" % (stage, runner.first_error_line(err_s))
        return ("CRASH" if crashed else "COMPILE_FAIL"), runner.short(reason), err_s
    rc, _, err, _ = runner.run_cmd(["as", "--64", "-o", obj, asm], 60)
    if rc != 0:
        return "ASM_FAIL", runner.short(runner.first_error_line(runner.text(err))), runner.text(err)
    rc, _, err, _ = runner.run_cmd(["gcc", "-no-pie", "-o", base, obj, "-lm"], 60)
    if rc != 0:
        return "LINK_FAIL", runner.short(runner.first_error_line(runner.text(err))), runner.text(err)
    return None, "", err_s


def run_seed(seed, cc_argv, inc, extra_args):
    os.makedirs(WORK, exist_ok=True)
    src = os.path.join(WORK, "%d.c" % seed)
    args = CSMITH_ARGS + extra_args + ["--seed", str(seed), "-o", src]
    rc, _, err, to = runner.run_cmd([CSMITH] + args, GEN_TIMEOUT)
    if rc != 0 or to:
        return {"seed": seed, "kind": "SKIP", "reason": "csmith failed"}
    meta = {"seed": seed, "csmith_args": args[:-2], "cc": shlex.join(cc_argv)}

    ref = os.path.join(WORK, "%d.gcc" % seed)
    kind, reason, _ = build_with(["gcc"], src, ref, inc, split=False)
    if kind:
        return dict(meta, kind="SKIP", reason=reason)
    rc, out, _, to = runner.run_cmd([ref], RUN_TIMEOUT)
    want = checksum_line(out)
    if to or rc != 0 or want is None:
        return dict(meta, kind="SKIP", reason="gcc binary timeout/failed")
    meta["gcc_output"] = want

    exe = os.path.join(WORK, "%d.cc" % seed)
    kind, reason, err_s = build_with(cc_argv, src, exe, inc, split=True)
    if kind:
        return dict(meta, kind=kind, reason=reason, cc_stderr=err_s[-3000:])
    rc, out, err, to = runner.run_cmd([exe], RUN_TIMEOUT)
    got = checksum_line(out)
    meta["cc_output"] = got
    if to:
        return dict(meta, kind="TIMEOUT", reason="run >%ds" % RUN_TIMEOUT)
    if rc != 0:
        why = ("killed by %s" % runner.signame(rc)) if rc < 0 else "exit %d" % rc
        return dict(meta, kind="RUN_FAIL", reason=why, cc_stderr=runner.text(err)[-3000:])
    if got != want:
        return dict(meta, kind="MISMATCH", reason="gcc %r vs cc %r" % (want, got))
    return dict(meta, kind="OK", reason="")


def cleanup(seed, keep_src):
    for p in glob.glob(os.path.join(WORK, "%d.*" % seed)):
        if keep_src and p.endswith(".c"):
            continue
        os.remove(p)


def parse_seeds(s):
    a, _, b = s.partition(":")
    return range(int(a), int(b or a) + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cc", default="python3 -m hcc")
    ap.add_argument("--seeds", default="1:20", help="inclusive range A:B (default 1:20)")
    ap.add_argument("--jobs", "-j", type=int, default=4)
    ap.add_argument("--json", help="write {seed: {kind, reason, ...}}")
    ap.add_argument("--csmith-arg", action="append", default=[],
                    help="extra csmith argument (repeatable), e.g. --csmith-arg=--no-bitfields")
    ap.add_argument("--keep", action="store_true", help="keep tests/csmith/work/<seed>.c for all seeds")
    args = ap.parse_args()
    if not os.path.exists(CSMITH):
        sys.exit("csmith not built: run tests/csmith/build.sh")
    inc = csmith_include()
    cc_argv = shlex.split(args.cc)
    os.makedirs(FAILURES, exist_ok=True)
    counts, results = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as ex:
        futs = {ex.submit(run_seed, s, cc_argv, inc, args.csmith_arg): s for s in parse_seeds(args.seeds)}
        for fut in concurrent.futures.as_completed(futs):
            seed = futs[fut]
            r = fut.result()
            results[seed] = r
            counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            print(("%s %d %s" % (r["kind"], seed, r["reason"])).rstrip(), flush=True)
            failed = r["kind"] not in ("OK", "SKIP")
            if failed:
                src = os.path.join(WORK, "%d.c" % seed)
                with open(src) as f, open(os.path.join(FAILURES, "%d.c" % seed), "w") as g:
                    g.write(f.read())
                with open(os.path.join(FAILURES, "%d.json" % seed), "w") as g:
                    json.dump(dict(r, include=inc), g, indent=1)
            cleanup(seed, keep_src=args.keep)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({str(k): results[k] for k in sorted(results)}, f, indent=1)
    print("SUMMARY seeds=%s %s" % (args.seeds, " ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    return 0 if set(counts) <= {"OK", "SKIP"} else 1


if __name__ == "__main__":
    sys.exit(main())
