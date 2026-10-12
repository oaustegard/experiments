"""Driver for the `cjson` suite: cJSON 1.7.18 built by hcc, with its Unity unit tests.

Test ids:
  cjson/src/cJSON.c, cjson/src/cJSON_Utils.c, cjson/src/unity.c
        hcc compiles that one file; it is linked into every unit-test program that links it
        (the others get the gcc -O0 object) and the whole upstream test run (21 programs, every
        case) must report the same PASS/FAIL per case as an all-gcc build.
  cjson/tests/<prog>.c    hcc compiles that unit-test file (many #include ../cJSON.c, so this also
        compiles cJSON.c); the rest is gcc.  All its cases must PASS as with gcc.
  cjson/case/<prog>/<case>  one Unity test case (RUN_TEST) of an entirely hcc-built program:
        the "file:line:name:PASS" line must say PASS.  A program that does not build fails all
        of its cases with the build error.
  cjson/root/test.c       the demo program in the repository root, built by hcc, output compared.
"""
import os
import re

import common as C
import projlib as P

URL = "https://github.com/DaveGamble/cJSON.git"
REF = "v1.7.18"
COMMIT = "acc76239bee01d8e9c858ae2cab296704e52d916"
PROGS = ["parse_examples", "parse_number", "parse_hex4", "parse_string", "parse_array", "parse_object",
         "parse_value", "print_string", "print_number", "print_array", "print_object", "print_value",
         "misc_tests", "parse_with_opts", "compare_tests", "cjson_add", "readme_examples", "minify_tests",
         "json_patch_tests", "old_utils_tests", "misc_utils_tests"]
RUN_TIMEOUT = 60
LINE = re.compile(r"^(?:.*?):(\d+):(\w+):(PASS|FAIL|IGNORE)\b(.*)$")


def root():
    return P.fetch_git("cJSON", URL, REF, COMMIT)


def cases(r, prog):
    src = open(os.path.join(r, "tests", prog + ".c"), errors="replace").read()
    return re.findall(r"^\s*RUN_TEST\((\w+)\)", src, re.M)


def list_tests(suite):
    r = root()
    out = ["src/cJSON.c", "src/cJSON_Utils.c", "src/unity.c", "root/test.c"]
    out += ["tests/%s.c" % p for p in PROGS]
    out += ["case/%s/%s" % (p, c) for p in PROGS for c in cases(r, p)]
    return out


def project(R, suite, cc_argv):
    r = root()
    units = {"cJSON": os.path.join(r, "cJSON.c"), "cJSON_Utils": os.path.join(r, "cJSON_Utils.c"),
             "unity": os.path.join(r, "tests", "unity", "src", "unity.c"), "root_test": os.path.join(r, "test.c")}
    for p in PROGS:
        units["t_" + p] = os.path.join(r, "tests", p + ".c")
    flags = ["-I" + r, "-I" + os.path.join(r, "tests"), "-DENABLE_LOCALES"]
    return P.Project(R, suite, cc_argv, "cjson", units, flags)


def prog_units(r, prog):
    """Units to link for a unit-test program (the library is already inside the test when it includes cJSON.c)."""
    src = open(os.path.join(r, "tests", prog + ".c"), errors="replace").read()
    common = open(os.path.join(r, "tests", "common.h"), errors="replace").read()
    inc_json = '"../cJSON.c"' in src or ('"common.h"' in src and '"../cJSON.c"' in common)
    inc_utils = '"../cJSON_Utils.c"' in src
    units = ["t_" + prog, "unity"]
    if not inc_json:
        units.append("cJSON")
    if 'cJSON_Utils.h' in src and not inc_utils:
        units.append("cJSON_Utils")
    return units


def workdir(r, work, name):
    d = P.tmpdir(work, name)
    for sub in ("inputs", "json-patch-tests"):
        dst = os.path.join(d, sub)
        if not os.path.exists(dst):
            P.copy_tree(os.path.join(r, "tests", sub), dst)
    return d


def run_prog(R, exe, d):
    rc, out, err, to = R.run_cmd([exe], RUN_TIMEOUT, cwd=d)
    cases_ = {}
    for line in R.text(out).splitlines():
        m = LINE.match(line)
        if m:
            cases_[m.group(2)] = m.group(3)
    return {"rc": rc, "cases": cases_, "timeout": to, "tail": R.text(err)[-300:]}


def run_test(suite, rel, cc_argv, work, R):
    proj = project(R, suite, cc_argv)
    r = root()
    kind, _, name = rel.partition("/")
    if kind == "src":
        return run_src(R, proj, r, name, work)
    if kind == "tests":
        return run_tests_tu(R, proj, r, name[:-2], work)
    if kind == "root":
        return run_root(R, proj, r, work)
    prog, case = name.split("/")
    return run_case(R, proj, r, prog, case, work)


def reference(R, proj, r, prog, work):
    exe = proj.ref("prog-" + prog, prog_units(r, prog), ["-lm"])
    return P.ref_value("cjson-ref", [C.file_sha(exe)], lambda: run_prog(R, exe, workdir(r, work, "ref-" + prog)))


def check(R, got, want, what=""):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "run >%ds" % RUN_TIMEOUT)
    if got["rc"] < 0:
        return R.Result("FAIL", "run", R.short("killed by %s %s" % (R.signame(got["rc"]), got["tail"])))
    bad = sorted(c for c, s in want["cases"].items() if got["cases"].get(c) != s)
    if bad:
        return R.Result("FAIL", "compare", R.short("%s: %s %s expected %s" % (what or bad[0], bad[0], got["cases"].get(bad[0], "missing"), want["cases"][bad[0]])))
    if got["rc"] != want["rc"]:
        return R.Result("FAIL", "compare", "exit %d, expected %d" % (got["rc"], want["rc"]))
    return R.Result("PASS")


def run_programs(R, proj, r, hcc_units, progs, work):
    for prog in progs:
        units = prog_units(r, prog)
        hit = [u for u in units if u in hcc_units]
        if not hit:
            continue
        exe = os.path.join(work, "p-" + prog)
        f = proj.mixed(exe, units, hcc_units, ["-lm"])
        if f:
            return f.result(R, "%s: " % prog)
        res = check(R, run_prog(R, exe, workdir(r, work, "got-" + prog)), reference(R, proj, r, prog, work), prog)
        if res.status != "PASS":
            return res
    return R.Result("PASS")


def run_src(R, proj, r, name, work):
    unit = {"cJSON.c": "cJSON", "cJSON_Utils.c": "cJSON_Utils", "unity.c": "unity"}[name]
    progs = [p for p in PROGS if unit in prog_units(r, p)]
    if not progs:
        f = proj.hobj(unit)[1]
        return f.result(R) if f else R.Result("PASS")
    return run_programs(R, proj, r, [unit], progs, work)


def run_tests_tu(R, proj, r, prog, work):
    return run_programs(R, proj, r, ["t_" + prog], [prog], work)


def run_root(R, proj, r, work):
    names = ["root_test", "cJSON"]
    out = os.path.join(work, "root_test")
    f = proj.mixed(out, names, ["root_test", "cJSON"], ["-lm"])
    if f:
        return f.result(R)
    ref = proj.ref("root_test", names, ["-lm"])
    want = P.ref_value("cjson-root", [C.file_sha(ref)], lambda: P.run(R, [ref], P.tmpdir(work, "ref"), RUN_TIMEOUT))
    return C.compare_runs(R, P.run(R, [out], P.tmpdir(work, "got"), RUN_TIMEOUT), want)


def run_case(R, proj, r, prog, case, work):
    units = prog_units(r, prog)
    exe, fail = proj.full("prog-" + prog, units, ["-lm"])
    if fail:
        return fail.result(R)
    want = reference(R, proj, r, prog, work)
    got = run_prog(R, exe, workdir(r, work, "got"))
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "run >%ds" % RUN_TIMEOUT)
    status = got["cases"].get(case)
    if status is None:
        why = "killed by %s" % R.signame(got["rc"]) if got["rc"] < 0 else "no result line (exit %d)" % got["rc"]
        return R.Result("FAIL", "run", R.short("%s %s" % (why, got["tail"])))
    if status != want["cases"].get(case):
        return R.Result("FAIL", "compare", "%s: %s, gcc build says %s" % (case, status, want["cases"].get(case)))
    return R.Result("PASS")
