#!/usr/bin/env python3
"""Turn csmith differential results into the `csmith` test suite (tests/suites/csmith).

    python3 tests/csmith/collect.py triage  --seeds-from tests/csmith/failures [--cc CC]
    python3 tests/csmith/collect.py reduce  [--cap 180] [--n 2]
    python3 tests/csmith/collect.py build   [--pass-count 200] [--max-per-signature 5]

triage   re-runs every saved failure with the current compiler and records which still fail
         (tests/cache/csmith/triage.json); fixed ones become regression-guard candidates.
reduce   shrinks each still-failing program with cvise (hard time cap per program; the best
         program found so far is kept when the cap hits, the original when nothing was
         found) into tests/cache/csmith/reduced/<seed>.c.
build    writes tests/suites/csmith: include/ (the csmith runtime headers), fail/<sig>_<seed>.c
         with .expected (stdout of gcc -O0), at most N per error signature, and pass/<seed>.c
         regression programs (OK seeds, plus fixed failures).

Generated tests are self-contained C files; the expected stdout and exit code come from a
gcc -O0 build of the very file in the suite (re-derived in `build`, never taken on trust).
"""
import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
sys.path.insert(0, HERE)
import diff  # noqa: E402
import run as runner  # noqa: E402

CACHE = os.path.join(TESTS, "cache", "csmith")
SUITE = os.path.join(TESTS, "suites", "csmith")


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


# --------------------------------------------------------------- triage

def check_file(cc, inc, src, work):
    """Like diff.run_seed on an already generated program: returns (kind, reason)."""
    ref = os.path.join(work, "ref")
    kind, reason, _ = diff.build_with(["gcc"], src, ref, inc, split=False)
    if kind:
        return "SKIP", reason
    rc, out, _, to = runner.run_cmd([ref], 40)
    want = diff.checksum_line(out)
    if to or rc != 0 or want is None:
        return "SKIP", "gcc binary timeout/failed"
    exe = os.path.join(work, "cc")
    kind, reason, _ = diff.build_with(cc, src, exe, inc, split=True)
    if kind:
        return kind, reason
    rc, out, _, to = runner.run_cmd([exe], diff.RUN_TIMEOUT)
    if to:
        return "TIMEOUT", "run >%ds" % diff.RUN_TIMEOUT
    if rc != 0:
        return "RUN_FAIL", ("killed by %s" % runner.signame(rc)) if rc < 0 else "exit %d" % rc
    got = diff.checksum_line(out)
    if got != want:
        return "MISMATCH", "gcc %r vs cc %r" % (want, got)
    return "OK", ""


def triage(args):
    cc = shlex.split(args.cc)
    inc = diff.csmith_include()
    seeds = sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(os.path.join(args.seeds_from, "*.json")))
    path = os.path.join(CACHE, "triage.json")
    res = load(path, {})
    work = tempfile.mkdtemp(prefix="triage-")
    for s in seeds:
        src = os.path.join(args.seeds_from, "%d.c" % s)
        kind, reason = check_file(cc, inc, src, work)
        res[str(s)] = {"kind": kind, "reason": reason, "src": src}
        print(s, kind, reason[:90], flush=True)
    shutil.rmtree(work, ignore_errors=True)
    save(path, res)


# --------------------------------------------------------------- reduce

def check_cmd(args, kind, reason):
    me = [sys.executable, os.path.join(TESTS, "reduce.py"), "--check", "--cc", args.cc, "-I", diff.csmith_include()]
    if kind in ("CRASH", "COMPILE_FAIL"):
        m = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", ".*", reason.split("error:", 1)[-1].strip())
        me += ["--compile-fail=" + re.escape(m).replace(r"\.\*", ".*")[:120]]
    else:
        me += ["--mismatch"]
    return me


def reduce_one(args, seed, kind, reason, src):
    out = os.path.join(CACHE, "reduced", "%d.c" % seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    work = tempfile.mkdtemp(prefix="cvise-")
    try:
        shutil.copy(src, os.path.join(work, "repro.c"))
        script = os.path.join(work, "interesting.sh")
        with open(script, "w") as f:
            f.write("#!/bin/sh\nexec %s repro.c\n" % shlex.join(check_cmd(args, kind, reason)))
        os.chmod(script, 0o755)
        # sanity: the original must satisfy the predicate
        if subprocess.run([script], cwd=work).returncode != 0:
            return None
        subprocess.run(["timeout", "-s", "TERM", "-k", "10", str(args.cap), "cvise", "--n", str(args.n),
                        "--timeout", "120", "--tidy", script, "repro.c"],
                       cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final = os.path.join(work, "repro.c")
        if subprocess.run([script], cwd=work).returncode != 0:  # cvise killed mid-write?
            final = src
        shutil.copy(final, out)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def reduce_all(args):
    import time
    t_end = time.time() + args.budget if args.budget else None
    tri = load(os.path.join(CACHE, "triage.json"), {})
    done = load(os.path.join(CACHE, "reduced.json"), {})
    for seed, t in sorted(tri.items(), key=lambda kv: int(kv[0])):
        if t["kind"] in ("OK", "SKIP") or seed in done or not t.get("src"):
            continue
        if t_end and time.time() > t_end:
            print("budget used up; rerun to continue")
            break
        out = reduce_one(args, int(seed), t["kind"], t["reason"], t["src"])
        before = os.path.getsize(t["src"])
        after = os.path.getsize(out) if out else None
        done[seed] = {"kind": t["kind"], "reason": t["reason"], "reduced": out, "bytes_before": before, "bytes_after": after}
        print("seed %s %s: %s -> %s bytes" % (seed, t["kind"], before, after), flush=True)
        save(os.path.join(CACHE, "reduced.json"), done)


# ---------------------------------------------------------------- build

KEYWORDS = ("volatile", "union", "long", "unsigned", "short", "char", "signed", "float", "double", "struct", "static",
            "const", "goto", "switch", "for", "while", "do", "break", "continue", "return", "int8_t", "int16_t",
            "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t")
OPERATORS = ("<<", ">>", "&&", "||", "?", "%", "/", "^", "~", "!", "==", "<=", ">=", "++", "--", "+=", "-=", "*=", "|=", "&=",
             "^=", "<<=", ">>=", "%=", "/=", ":")


def signature(kind, reason, text):
    """Dedupe key.  Compile failures: stage + normalised message.  Miscompiles carry no
    message, so: the keywords/operators the *reduced* program still uses (what a reducer
    leaves behind is what the bug needs)."""
    if kind in ("CRASH", "COMPILE_FAIL", "ASM_FAIL", "LINK_FAIL"):
        m = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "N", reason)
        return kind.lower() + ": " + m[:80]
    body = re.sub(r"#include.*|//.*|/\*.*?\*/", "", text, flags=re.S)
    words = set(re.findall(r"[A-Za-z_]\w*", body))
    feats = [k for k in KEYWORDS if k in words] + [o for o in OPERATORS if o in body]
    # the two most telling features only, so near-identical repros collapse
    rank = ["volatile", "union", ":", "goto", "switch", "float", "double", "<<=", ">>=", "<<", ">>", "%", "/", "?", "long", "uint64_t", "int64_t", "short", "int16_t", "uint16_t", "char", "int8_t", "uint8_t", "unsigned", "struct"]
    top = [f for f in rank if f in feats][:2]
    return kind.lower() + ": " + "+".join(top or ["plain"])


def slug(sig):
    return re.sub(r"[^a-z0-9]+", "-", sig.lower()).strip("-")[:40]


def gcc_expected(src, inc, work):
    exe = os.path.join(work, "ref")
    rc, _, err, to = runner.run_cmd(["gcc", "-O0", "-w", "-I" + inc, "-o", exe, src, "-lm"], 120)
    if rc != 0:
        return None
    rc, out, _, to = runner.run_cmd([exe], 10)
    if rc != 0 or to:
        return None
    return out


def norm_reason(reason):
    return re.sub(r"'[^']*'|\"[^\"]*\"|0x[0-9a-fA-F]+|\d+", "N", reason)[:80]


def run_candidates(args):
    """Run every file in SUITE/_cand through tests/run.py with the compiler under test."""
    cfg = load(os.path.join(SUITE, "suite.json"), {})
    tmp_cfg = dict(cfg, glob=["_cand/*.c"])
    save(os.path.join(SUITE, "suite.json"), tmp_cfg)
    out = os.path.join(CACHE, "cand_results.json")
    try:
        runner.main(["--suite", "csmith", "--jobs", str(args.n), "--cc", args.cc, "--json", out, "-q"])
    finally:
        save(os.path.join(SUITE, "suite.json"), cfg)
    return load(out, {})


def build(args):
    inc = diff.csmith_include()
    os.makedirs(SUITE, exist_ok=True)
    cfg = {
        "description": "csmith programs (github.com/csmith-project/csmith, built by tests/csmith/build.sh), differential vs gcc -O0. "
                       "fail/: reduced repros of hcc failures found by tests/csmith/diff.py (seeds 1..2000 plus wider csmith configurations), "
                       "at most %d per error signature; the file name is the signature slug plus the seed. "
                       "fixed/: programs that failed when found and pass now (regression guard). pass/: %d programs hcc always compiled correctly. "
                       "Pass = exit 0 and stdout == .expected (stdout of a gcc -O0 build). Collected by tests/csmith/collect.py."
                       % (args.max_per_signature, args.pass_count),
        "glob": ["fail/*.c", "fixed/*.c", "pass/*.c"],
        "cflags": ["-w", "-I{suite}/include"],
        "gcc_cflags": ["-O0"],
    }
    for sub in ("fail", "fixed", "pass", "include", "_cand"):
        shutil.rmtree(os.path.join(SUITE, sub), ignore_errors=True)
        os.makedirs(os.path.join(SUITE, sub))
    save(os.path.join(SUITE, "suite.json"), cfg)
    for p in glob.glob(os.path.join(inc, "*.h")):
        shutil.copy(p, os.path.join(SUITE, "include"))
    work = tempfile.mkdtemp(prefix="csmith-build-")
    red = load(os.path.join(CACHE, "reduced.json"), {})
    tri = load(os.path.join(CACHE, "triage.json"), {})

    # 1. candidates: (name, source path, kind, original reason, origin)
    cands = {}
    for seed, t in sorted(tri.items(), key=lambda kv: int(kv[0])):
        if not t.get("src"):
            continue
        r = red.get(seed)
        if r and r.get("reduced"):
            cands["r%s" % seed] = (r["reduced"], r["kind"], r["reason"])
        # unreduced originals: fixed ones (regression guard), and build/link failures (no UB question).
        # A miscompile that could not be reduced failed the reducer's UB check (sanitizers, strict
        # warnings) on the original, so its expected output cannot be trusted: dropped.
        if t["kind"] == "OK":
            cands["o%s" % seed] = (t["src"], (r or {}).get("kind", "MISMATCH"), t["reason"])
        elif not (r and r.get("reduced")) and t["kind"] in ("LINK_FAIL", "ASM_FAIL", "COMPILE_FAIL", "CRASH"):
            cands["o%s" % seed] = (t["src"], t["kind"], t["reason"])
    for name, (src, kind, reason) in cands.items():
        exp = gcc_expected(src, inc, work)
        if exp is None:
            print("%s: gcc reference unusable, dropped" % name)
            continue
        shutil.copy(src, os.path.join(SUITE, "_cand", name + ".c"))
        with open(os.path.join(SUITE, "_cand", name + ".c.expected"), "wb") as f:
            f.write(exp)

    # 2. classify against the compiler under test
    res = run_candidates(args)
    per_sig, sigs, n_fail, n_fixed = {}, {}, 0, 0
    for tid, r in sorted(res.items()):
        name = os.path.basename(tid)[:-2]
        src = os.path.join(SUITE, "_cand", name + ".c")
        _, kind, reason = cands[name]
        if r["status"] == "PASS":
            if name.startswith("r") or True:
                dst = os.path.join(SUITE, "fixed", name + ".c")
                shutil.move(src, dst)
                shutil.move(src + ".expected", dst + ".expected")
                n_fixed += 1
            continue
        with open(src) as f:
            text = f.read()
        if r["stage"] in ("compare", "run", "timeout", None):
            sig = signature({"run": "RUN_FAIL", "timeout": "TIMEOUT"}.get(r["stage"], "MISMATCH"), "", text)
        else:
            sig = "%s: %s" % (r["stage"], norm_reason(r["reason"]))
        if per_sig.get(sig, 0) >= args.max_per_signature:
            continue
        # prefer the reduced version of a seed over its original
        if name.startswith("o") and ("r" + name[1:]) in res and res["r" + name[1:]]["status"] != "PASS":
            continue
        dst = os.path.join(SUITE, "fail", "%s_%s.c" % (slug(sig), name[1:] if name[0] in "ro" else name))
        shutil.move(src, dst)
        shutil.move(src + ".expected", dst + ".expected")
        per_sig[sig] = per_sig.get(sig, 0) + 1
        sigs[sig] = sigs.get(sig, 0) + 1
        n_fail += 1

    # 3. regression guard: programs that always passed
    pass_seeds = load(os.path.join(CACHE, "pass_seeds.json"), [])
    n_pass = 0
    for seed in pass_seeds[:args.pass_count]:
        src = os.path.join(work, "%d.c" % seed)
        rc, _, _, _ = runner.run_cmd([diff.CSMITH] + diff.CSMITH_ARGS + ["--seed", str(seed), "-o", src], diff.GEN_TIMEOUT)
        if rc != 0:
            continue
        exp = gcc_expected(src, inc, work)
        if exp is None:
            continue
        dst = os.path.join(SUITE, "pass", "s%d.c" % seed)
        shutil.copy(src, dst)
        with open(dst + ".expected", "wb") as f:
            f.write(exp)
        n_pass += 1
    shutil.rmtree(os.path.join(SUITE, "_cand"), ignore_errors=True)
    print("fail tests: %d (%d signatures), fixed: %d, pass: %d" % (n_fail, len(sigs), n_fixed, n_pass))
    for sig, n in sorted(sigs.items()):
        print("  %3d %s" % (n, sig))
    shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("triage", "reduce", "build"))
    ap.add_argument("--cc", default="python3 -m hcc")
    ap.add_argument("--seeds-from", default=os.path.join(HERE, "failures"))
    ap.add_argument("--cap", type=int, default=180, help="seconds of cvise per program")
    ap.add_argument("--budget", type=int, default=0, help="reduce: stop starting new programs after this many seconds (rerun resumes)")
    ap.add_argument("--n", type=int, default=2, help="cvise parallelism")
    ap.add_argument("--pass-count", type=int, default=200)
    ap.add_argument("--max-per-signature", type=int, default=5)
    args = ap.parse_args()
    {"triage": triage, "reduce": reduce_all, "build": build}[args.step](args)


if __name__ == "__main__":
    main()
