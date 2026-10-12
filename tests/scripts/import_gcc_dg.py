#!/usr/bin/env python3
"""(Re)build the gcc-dg* suites from gcc/testsuite/gcc.dg.

    python3 tests/scripts/import_gcc_dg.py [tests/tools/src/gcc] [--only gcc-dg]

(see import_gcc_compile.py for fetching the sparse GCC checkout.)

Run suites: every gcc.dg test whose applicable `dg-do` is `run` (dg selectors are
evaluated for x86_64-linux-gnu by scripts/dg.py).  Pass = exit 0, like gcc-torture; tests
with a dg-output get <t>.c.expected from the gcc -O0 binary's stdout instead (the dg-output
regex is not reproduced).  Directories covered: gcc.dg/*.c plus its torture, tree-ssa,
ipa, atomic, sso, tls, cpp, special, debug, charset, weak and rtl subdirectories (ids
`gcc-dg/<dir>/<name>.c`); asan/ubsan/tsan/gomp/graphite/vect/fixed-point/dfp/guality/lto/plugin
need runtimes or target features hcc is not meant to have.

Everything not usable lands in skip.txt with a reason: restricted to another target,
unsupported effective-target / option, dg-additional-sources (multi-file), dg-shouldfail,
dg-error, and whatever gcc -O0 -w itself fails (compile, link, run or output mismatch).
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dg  # noqa: E402

SUITES = {
    "gcc-dg": [".", "torture", "tree-ssa", "ipa", "atomic", "sso", "tls", "cpp", "special", "debug", "charset",
               "weak", "rtl"],
}
DESC = ("gcc/testsuite/gcc.dg tests whose applicable dg-do is `run` (%s). Pass = exit 0 (and stdout equal to the "
        "gcc -O0 binary's where the test has a dg-output). Per-test flags from dg-options in <test>.c.flags; skip.txt "
        "lists inapplicable tests and tests gcc -O0 itself fails. Regenerate with tests/scripts/import_gcc_dg.py.")


def classify(text):
    """Returns (flags, skip_reason or None, is_candidate)."""
    dirs = dg.directives(text)
    if not re.search(r"dg-do\s+\"?run", text):
        return [], None, False
    try:
        kind = dg.do_kind(dirs)
    except dg.Unknown as e:
        return [], "dg-do guard uses unknown feature %s" % e, True
    flags, opt_reasons = dg.options(dirs)
    flags = ["-std=gnu2x" if f == "-std=gnu23" else "-std=c2x" if f == "-std=c23" else f for f in flags]
    names = {n for n, _ in dirs}
    reason = None
    if kind != "run":
        reason = "dg-do run restricted to another target"
    elif "dg-additional-sources" in names:
        reason = "multi-file test (dg-additional-sources)"
    elif "dg-shouldfail" in names:
        reason = "dg-shouldfail: expected to fail at run time"
    elif "dg-error" in names:
        reason = "dg-error: gcc must reject this test"
    elif "dg-output-file" in names or "dg-output-file-" in names:
        reason = "dg-output-file (expected output in another file)"
    elif "dg-timeout" in names:
        reason = "dg-timeout (slow test)"
    reason = reason or dg.skip_if(dirs, flags) or (opt_reasons[0] if opt_reasons else None)
    return flags, reason, True


def gcc_reference(path, flags):
    """stdout of the gcc -O0 binary (run twice; None if nondeterministic or failing)."""
    exe = path + ".ref.exe"
    r = subprocess.run(["gcc", "-O0", "-w", "-no-pie", "-o", exe] + flags + [path, "-lm"],
                       capture_output=True, text=True)
    if r.returncode:
        return None
    outs = []
    try:
        for _ in range(2):
            p = subprocess.run([exe], capture_output=True, stdin=subprocess.DEVNULL, timeout=10)
            outs.append(p.stdout)
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.path.exists(exe) and os.remove(exe)
    return outs[0] if outs[0] == outs[1] else None


def build_suite(name, dirs_, src_root):
    root = os.path.join(src_root, "gcc", "testsuite", "gcc.dg")
    dest = os.path.join(TESTS, "suites", name)
    os.makedirs(dest, exist_ok=True)
    for dp, _, fn in os.walk(dest):
        for f in fn:
            if f.endswith((".c", ".h", ".flags", ".expected")) or f == "skip.txt":
                os.remove(os.path.join(dp, f))
    skips, n_cand = {}, 0
    globs = []
    for d in dirs_:
        src = root if d == "." else os.path.join(root, d)
        out = dest if d == "." else os.path.join(dest, d)
        os.makedirs(out, exist_ok=True)
        rel_prefix = "" if d == "." else d + "/"
        globs.append(rel_prefix + "*.c")
        for f in sorted(os.listdir(src)):
            p = os.path.join(src, f)
            if not os.path.isfile(p):
                continue
            if f.endswith((".h", ".inc")):
                shutil.copy(p, out)
                continue
            if not f.endswith(".c"):
                continue
            text = open(p, errors="replace").read()
            flags, reason, cand = classify(text)
            if not cand:
                continue
            n_cand += 1
            shutil.copy(p, out)
            dst = os.path.join(out, f)
            if flags:
                with open(dst + ".flags", "w") as fh:
                    fh.write(" ".join(flags) + "\n")
            if reason:
                skips[rel_prefix + f] = reason
            elif re.search(r"\bdg-output\b", text):
                ref = gcc_reference(dst, flags)
                if ref is None:
                    skips[rel_prefix + f] = "dg-output: gcc -O0 reference output unavailable or nondeterministic"
                else:
                    with open(dst + ".expected", "wb") as fh:
                        fh.write(ref)
    cfg = {"description": DESC % ", ".join(d if d != "." else "top level" for d in dirs_),
           "glob": globs, "cflags": ["-w"]}
    with open(os.path.join(dest, "suite.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    write_skips(dest, skips)
    # Everything left must pass with gcc -O0 -w.
    out = os.path.join(dest, ".gcc-baseline.json")
    subprocess.run([sys.executable, os.path.join(TESTS, "run.py"), "--suite", name,
                    "--cc", "gcc -O0", "--jobs", "2", "--json", out, "-q"], check=False)
    for tid, r in json.load(open(out)).items():
        if r["status"] != "PASS":
            skips[tid.split("/", 1)[1]] = "gcc -O0 -w fails (%s: %s)" % (r["stage"], r["reason"][:80])
    os.remove(out)
    write_skips(dest, skips)
    rev = subprocess.run(["git", "-C", src_root, "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(dest, "SOURCE"), "w") as fh:
        fh.write("https://github.com/gcc-mirror/gcc %s gcc/testsuite/gcc.dg (%s)\n" % (rev, ", ".join(dirs_)))
    print("%s: %d dg-do run tests, %d skipped, %d active" % (name, n_cand, len(skips), n_cand - len(skips)))


def write_skips(dest, skips):
    with open(os.path.join(dest, "skip.txt"), "w") as fh:
        fh.write("# Generated by tests/scripts/import_gcc_dg.py: <test>  # reason\n")
        for f in sorted(skips):
            fh.write("%s  # %s\n" % (f, skips[f].replace("\n", " ")))


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1]
        del args[i:i + 2]
    src_root = args[0] if args else os.path.join(TESTS, "tools", "src", "gcc")
    for name, dirs_ in SUITES.items():
        if only and name != only:
            continue
        build_suite(name, dirs_, src_root)


if __name__ == "__main__":
    main()
