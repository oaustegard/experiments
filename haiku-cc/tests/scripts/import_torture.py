#!/usr/bin/env python3
"""(Re)build tests/suites/gcc-torture from a GCC checkout.

    git clone --depth 1 --filter=blob:none --sparse https://github.com/gcc-mirror/gcc.git SRC
    (cd SRC && git sparse-checkout set gcc/testsuite/gcc.c-torture/execute)
    python3 tests/scripts/import_torture.py SRC

Steps: copy top-level execute/*.c and *.h (not ieee/ or builtins/); turn
supported dg-options into <test>.c.flags; write skip.txt with a reason per
skipped test (unsupported dg-require-effective-target, GCC-only extensions
that clang rejects, tests gcc -O0 itself fails); write tags.json with
feature tags per test so the swarm can prioritise.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import concurrent.futures

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
DEST = os.path.join(TESTS, "suites", "gcc-torture")

SKIP_TARGETS = {
    "trampolines": "nested functions with trampolines (GCC-only)",
    "untyped_assembly": "__builtin_apply/untyped calls (GCC-only)",
    "run_expensive_tests": "expensive test",
    "dfp": "decimal floating point", "dfprt": "decimal floating point",
}
# dg-options we pass on; everything else (optimisation/target flags) is dropped.
KEEP_FLAGS = re.compile(r"^(-std=\S+|-fgnu89-inline|-fwrapv|-fno-strict-overflow|-funsigned-char|-fsigned-char)$")

TAGS = [
    ("vector", r"vector_size|__attribute__\s*\(\(\s*vector"),
    ("int128", r"__int128"),
    ("complex", r"_Complex|__complex__|\bcomplex\b"),
    ("computed-goto", r"&&\s*[A-Za-z_]\w*\s*[,;}\)]|goto\s*\*"),
    ("inline-asm", r"\b(__asm__|asm)\s*(volatile|__volatile__)?\s*\("),
    ("builtin", r"__builtin_(?!abort|exit|memcpy|memset|strlen|strcmp|memcmp)"),
    ("attribute", r"__attribute__"),
    ("bitfield", r":\s*\d+\s*;"),
    ("float", r"\b(float|double)\b"),
    ("varargs", r"\.\.\.\s*\)|va_arg"),
    ("struct-return", r"struct\s+\w+\s+\w+\s*\([^)]*\)\s*\{"),
    ("knr", r"^\w[\w\s\*]*\(\s*\w+(\s*,\s*\w+)*\s*\)\s*\n\s*(int|char|long|unsigned|struct|short|double|float)\b"),
    ("vla", r"\[\s*[a-z_]\w*\s*\]\s*;"),
    ("statement-expr", r"\(\{"),
    ("typeof", r"\b(__typeof__|typeof)\b"),
    ("label-values", r"&&[a-zA-Z_]"),
]


def classify(src_text):
    reasons, flags = [], []
    for m in re.finditer(r"dg-require-effective-target\s+(\w+)", src_text):
        if m.group(1) in SKIP_TARGETS:
            reasons.append("requires %s: %s" % (m.group(1), SKIP_TARGETS[m.group(1)]))
    if re.search(r'dg-skip-if "requires higher optimization" \{ \*-\*-\* \}', src_text):
        reasons.append("requires optimisation")
    for m in re.finditer(r"dg-(?:additional-)?options\s+(?:\{\s*)?\"([^\"]*)\"\s*(\{[^}]*target)?", src_text):
        if m.group(2):  # target-specific options: ignore
            continue
        flags += [f for f in m.group(1).split() if KEEP_FLAGS.match(f)]
    tags = [name for name, rx in TAGS if re.search(rx, src_text, re.M)]
    return reasons, flags, tags


def clang_rejects(path, flags):
    cmd = ["clang", "-fsyntax-only", "-w", "-Wno-error", "-ferror-limit=1",
           "-Wno-error=implicit-function-declaration", "-Wno-error=implicit-int",
           "-Wno-error=int-conversion", "-Wno-error=incompatible-function-pointer-types",
           "-Wno-error=return-type"] + flags + [path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        return None
    errs = [l for l in r.stderr.splitlines() if "error:" in l]
    return (errs[0].split("error:", 1)[1].strip() if errs else "clang error")[:100]


def main():
    src_root = sys.argv[1]
    exe = os.path.join(src_root, "gcc", "testsuite", "gcc.c-torture", "execute")
    if os.path.isdir(DEST):
        for f in os.listdir(DEST):
            if f.endswith((".c", ".h", ".flags")):
                os.remove(os.path.join(DEST, f))
    os.makedirs(DEST, exist_ok=True)
    skips, tags_all = {}, {}
    names = sorted(f for f in os.listdir(exe) if f.endswith((".c", ".h")))
    for f in names:
        shutil.copy(os.path.join(exe, f), DEST)
    tests = [f for f in names if f.endswith(".c")]
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        def one(f):
            text = open(os.path.join(DEST, f), errors="replace").read()
            reasons, flags, tags = classify(text)
            if flags:
                with open(os.path.join(DEST, f + ".flags"), "w") as fh:
                    fh.write(" ".join(flags) + "\n")
            if not reasons:
                err = clang_rejects(os.path.join(DEST, f), flags)
                if err:
                    reasons.append("GCC-only extension (clang: %s)" % err)
            return f, reasons, tags
        for f, reasons, tags in ex.map(one, tests):
            tags_all[f] = tags
            if reasons:
                skips[f] = reasons[0]
    write_skips(skips)
    # Now run everything not yet skipped with gcc -O0 and skip what gcc fails.
    out = os.path.join(DEST, ".gcc-baseline.json")
    subprocess.run([sys.executable, os.path.join(TESTS, "run.py"), "--suite", "gcc-torture",
                    "--cc", "gcc -O0", "--json", out, "-q"], check=False)
    for tid, r in json.load(open(out)).items():
        if r["status"] != "PASS":
            skips[tid.split("/", 1)[1]] = "fails with gcc -O0 (%s: %s)" % (r["stage"], r["reason"][:80])
    os.remove(out)
    write_skips(skips)
    with open(os.path.join(DEST, "tags.json"), "w") as fh:
        json.dump(tags_all, fh, indent=0, sort_keys=True)
    rev = subprocess.run(["git", "-C", src_root, "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(DEST, "SOURCE"), "w") as fh:
        fh.write("https://github.com/gcc-mirror/gcc %s gcc/testsuite/gcc.c-torture/execute\n" % rev)
    print("imported %d tests, %d skipped" % (len(tests), len(skips)))


def write_skips(skips):
    with open(os.path.join(DEST, "skip.txt"), "w") as fh:
        fh.write("# Generated by tests/scripts/import_torture.py: <test>  # reason\n")
        for f in sorted(skips):
            fh.write("%s  # %s\n" % (f, skips[f]))


if __name__ == "__main__":
    main()
