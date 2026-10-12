"""Driver for rejection suites (reject-gcc-dg).

Test ids: <suite>/<dir>/<file>.c.  The file is invalid C that gcc rejects with an error at -w and default
dialect (checked by the importer, which also stores gcc's first diagnostic in <t>.c.gccerror).  PASS = the compiler
under test exits with status 1 (a diagnostic, no output file needed).  Accepting the program (exit 0) is a FAIL
with stage compare, a crash (traceback, exit 70, signal) a CRASH.  `--cc gcc` passes by construction.
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
    gccerr = ""
    if os.path.exists(src + ".gccerror"):
        with open(src + ".gccerror") as f:
            gccerr = f.read().strip()
    asm = os.path.join(work, "t.s")
    rc, out, err, to = R.run_cmd(cc_argv + cflags + ["-S", "-o", asm, src], COMPILE_TIMEOUT, cwd=work, env=R.cc_env())
    err_s = R.text(err)
    if to:
        return R.Result("TIMEOUT", "timeout", "compile >%ds" % COMPILE_TIMEOUT)
    if R.is_compiler_crash(rc, err_s):
        return R.Result("CRASH", R.stage_from_stderr(err_s) or "parse", R.short("compiler crashed on an invalid program: %s" % R.first_error_line(err_s)), err_s)
    if rc == 0:
        return R.Result("FAIL", "compare", R.short("accepted an invalid program; gcc says: %s" % gccerr), "")
    return R.Result("PASS")
