"""Driver for compile-only suites (gcc-compile).

Test ids: <suite>/<file>.c.  PASS = `<cc> <cflags> -S` succeeds and `as --64` accepts
the assembly.  Nothing is linked or run.  Per-test extra flags come from <t>.c.flags,
exactly as in the single-file runner.  `--cc gcc` must pass every active test: the
importer (tests/scripts/import_gcc_compile.py) puts the tests gcc -O0 -w itself rejects
into skip.txt.
"""
import glob
import os
import shlex

import common as C

COMPILE_TIMEOUT = 60


def list_tests(suite):
    out = []
    for pat in suite.cfg.get("glob", ["*.c"]):
        out += [os.path.relpath(p, suite.dir) for p in glob.glob(os.path.join(suite.dir, pat), recursive=True)]
    return out


def run_test(suite, rel, cc_argv, work, R):
    src = os.path.join(suite.dir, rel)
    cflags = suite.cflags(cc_argv)
    if os.path.exists(src + ".flags"):
        with open(src + ".flags") as f:
            cflags += shlex.split(f.read())
    obj = os.path.join(work, "t.o")
    fail = C.compile_c(R, cc_argv, cflags, src, obj, work, COMPILE_TIMEOUT)
    return fail.result(R) if fail else R.Result("PASS")
