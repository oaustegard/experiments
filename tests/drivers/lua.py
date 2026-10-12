"""Driver for the `lua` suite: Lua 5.4.7 built by hcc.

Test ids:
  lua/src/<f>.c        hcc compiles this one file; the object is linked with the
                       gcc -O0 objects of all other files into a lua (or luac for
                       luac.c); a short smoke script (suite/smoke.lua) must give the
                       same stdout/exit code as with a fully gcc-built lua.
  lua/testes/<f>.lua   an official lua-5.4.7-tests file, run by a lua built
                       entirely by hcc, in the tests' portable mode
                       (-e"_U=true;_port=true;_soft=true;_nomsg=true"), from a
                       private copy of testes/.  Expected result: what a gcc-built
                       lua gives for the same command (exit code + stdout).
Only the C files are compiled by hcc; linking and assembling use the host tools.
hcc objects are cached per hcc source hash (tests/drivers/common.py), so the
whole lua is compiled once per hcc revision however many tests run.
If lvm.c fails with the computed-goto dispatch (the default under gcc), the
whole-program build retries it with -DLUA_USE_JUMPTABLE=0, so the official tests
still exercise the rest of hcc; the lvm.c test keeps the default configuration.
"""
import glob
import os
import re
import shutil

import common as C

LDFLAGS = ["-lm"]
RUN_TIMEOUT = 300
SMOKE_TIMEOUT = 60
NORMALISE = [
    (r"0x[0-9a-fA-F]+", "0xPTR"),
    (r"/tmp/lua_[A-Za-z0-9_]+", "/tmp/lua_X"),
    (r"test done on [0-9/]+, at [0-9:]+", "test done on DATE"),
    (r"\b\d+(\.\d+)?\s*(s|secs|seconds)\b", "TIME"),
    (r"random seeds: -?\d+, -?\d+", "random seeds: X, Y"),
    (r"short-circuit optimizations \(\d\)", "short-circuit optimizations (N)"),
    (r"in [0-9.]+ msec", "in T msec"),
    (r"with \d+ comparisons", "with N comparisons"),
    (r"random range in \d+ calls: .*", "random range in N calls"),
]
# Files whose standalone invocation needs the context all.lua provides.
EXTRA_E = {
    "big.lua": "local f = coroutine.wrap(assert(loadfile('big.lua'))); assert(f() == 'b'); assert(f() == 'a')",
    "attrib.lua": "assert(dofile('attrib.lua') == 27)",
    "locals.lua": "assert(dofile('locals.lua') == 5)",
    "events.lua": "assert(dofile('events.lua') == 12)",
    "verybig.lua": "assert(dofile('verybig.lua') == 10)",
    "calls.lua": "assert(dofile('calls.lua') == deep and deep)",
}
PORTABLE = "_U=true; _port=true; _soft=true; _nomsg=true"


def list_tests(suite):
    out = ["src/" + os.path.basename(p) for p in glob.glob(os.path.join(suite.dir, "src", "*.c"))]
    out += ["testes/" + os.path.basename(p) for p in glob.glob(os.path.join(suite.dir, "testes", "*.lua"))]
    return out


def lib_names(suite):
    return sorted(os.path.basename(p)[:-2] for p in glob.glob(os.path.join(suite.dir, "src", "*.c"))
                  if os.path.basename(p) not in ("lua.c", "luac.c"))


def src_hash(suite):
    return C.sha(*[C.file_sha(p) for p in sorted(glob.glob(os.path.join(suite.dir, "src", "*")))])


def gcc_objs(suite, flags):
    """name -> gcc -O0 object, for every src/*.c (including lua.c and luac.c)."""
    objs = {}
    for p in sorted(glob.glob(os.path.join(suite.dir, "src", "*.c"))):
        name = os.path.basename(p)[:-2]
        objs[name] = C.gcc_object(flags, p, "lua/" + name)
    return objs


def ref_program(R, suite, flags, which):
    """gcc-built `lua` or `luac`."""
    objs = gcc_objs(suite, flags)
    skip = "luac" if which == "lua" else "lua"
    use = [o for n, o in sorted(objs.items()) if n != skip]

    def build(out):
        return C.link(R, out, use, LDFLAGS, os.path.dirname(out))
    path, fail = C.cached_program(R, "lua-ref-" + which, [C.gcc_fingerprint(), src_hash(suite), " ".join(flags)], build)
    if fail:
        raise RuntimeError("reference %s failed to link: %s" % (which, fail.reason))
    return path


def make_workdir(suite, work, lua_path, with_tests=False):
    d = os.path.join(work, "t")
    if with_tests:
        shutil.copytree(os.path.join(suite.dir, "testes"), d)
    else:
        os.makedirs(d)
        shutil.copy(os.path.join(suite.dir, "smoke.lua"), d)
    if lua_path:
        shutil.copy(lua_path, os.path.join(d, "lua"))
    return d


def norm_run(r):
    r = dict(r)
    r["out"] = C.normalise(r["out"], NORMALISE)
    return r


# ------------------------------------------------------------ smoke (src/*.c)

def smoke_lua(R, lua_path, work, suite):
    d = make_workdir(suite, work, lua_path)
    return norm_run(C.run_and_capture(R, ["./lua", "smoke.lua"], SMOKE_TIMEOUT, d))


def smoke_luac(R, luac_path, ref_lua, work, suite):
    d = make_workdir(suite, work, ref_lua)
    shutil.copy(luac_path, os.path.join(d, "luac"))
    parts, rc = [], 0
    for argv in (["./luac", "-l", "-l", "smoke.lua"], ["./luac", "-p", "smoke.lua"],
                 ["./luac", "-s", "-o", "out.luac", "smoke.lua"], ["./lua", "out.luac"]):
        r = C.run_and_capture(R, argv, SMOKE_TIMEOUT, d)
        parts.append("$ %s\n%s" % (" ".join(argv), r["out"]))
        if r["rc"] != 0 or r["timeout"]:
            rc = r["rc"] or 1
            parts.append("[%s: exit %s]\n%s" % (argv[0], r["rc"], r["err"][-500:]))
            break
    return norm_run({"rc": rc, "out": "".join(parts), "err": "", "timeout": False})


def run_src_test(suite, rel, cc_argv, work, R):
    flags = suite.cflags(cc_argv)
    gflags = list(suite.cfg["cflags"])
    name = os.path.basename(rel)[:-2]
    obj, fail = C.cached_object(R, cc_argv, flags, os.path.join(suite.dir, rel), "lua/" + name)
    if fail:
        return fail.result(R)
    objs = gcc_objs(suite, gflags)
    ref_lua = ref_program(R, suite, gflags, "lua")
    exe = os.path.join(work, "prog")
    if name == "luac":
        use = [o for n, o in sorted(objs.items()) if n not in ("lua", "luac")] + [obj]
        f = C.link(R, exe, use, LDFLAGS, work)
        if f:
            return f.result(R)
        want = C.cached_value("lua-smoke-luac", [C.gcc_fingerprint(), src_hash(suite), C.file_sha(os.path.join(suite.dir, "smoke.lua")), C.file_sha(__file__)],
                              lambda: smoke_luac(R, ref_program(R, suite, gflags, "luac"), ref_lua,
                                                 _tmp(work, "ref"), suite))
        got = smoke_luac(R, exe, ref_lua, _tmp(work, "got"), suite)
        return C.compare_runs(R, got, want)
    use = [obj if n == name else o for n, o in sorted(objs.items()) if n != "luac"]
    if name not in objs or name == "lua":
        use = [o for n, o in sorted(objs.items()) if n not in ("luac", "lua")] + [obj]
    f = C.link(R, exe, use, LDFLAGS, work)
    if f:
        return f.result(R)
    want = C.cached_value("lua-smoke", [C.gcc_fingerprint(), src_hash(suite), C.file_sha(os.path.join(suite.dir, "smoke.lua")), C.file_sha(__file__)],
                          lambda: smoke_lua(R, ref_lua, _tmp(work, "ref"), suite))
    got = smoke_lua(R, exe, _tmp(work, "got"), suite)
    return C.compare_runs(R, got, want)


def _tmp(work, name):
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------- official tests

def hcc_lua(R, suite, cc_argv):
    """Build (cached) a lua with every file compiled by hcc.  Returns (path, None) or
    (None, Failure)."""
    flags = suite.cflags(cc_argv)

    def build(out):
        objs, first_fail = [], None
        for name in lib_names(suite) + ["lua"]:
            src = os.path.join(suite.dir, "src", name + ".c")
            obj, fail = C.cached_object(R, cc_argv, flags, src, "lua/" + name)
            if fail and name == "lvm":
                obj, fail2 = C.cached_object(R, cc_argv, flags + ["-DLUA_USE_JUMPTABLE=0"], src, "lua/lvm-nojt")
                if not fail2:
                    fail = None
            if fail:
                first_fail = first_fail or (name, fail)
            else:
                objs.append(obj)
        if first_fail:
            name, fail = first_fail
            return C.Failure(fail.status, fail.stage, "lua build: %s.c: %s" % (name, fail.reason), fail.log)
        return C.link(R, out, objs, LDFLAGS, os.path.dirname(out))
    return C.cached_program(R, "lua-hcc", [C.compiler_fingerprint(R, cc_argv), src_hash(suite), " ".join(flags)], build)


def lua_argv(name):
    snippet = EXTRA_E.get(name)
    argv = ["./lua", "-e" + PORTABLE]
    if snippet:
        argv += ["-e", snippet]
        if name == "big.lua":
            return argv
        return argv  # the snippet dofile()s the test itself
    return argv + [name]


def run_official(R, lua_path, work, suite, name, timeout):
    d = make_workdir(suite, work, lua_path, with_tests=True)
    return norm_run(C.run_and_capture(R, lua_argv(name), timeout, d))


def run_testes_test(suite, rel, cc_argv, work, R):
    name = os.path.basename(rel)
    gflags = list(suite.cfg["cflags"])
    ref_lua = ref_program(R, suite, gflags, "lua")
    key = [C.gcc_fingerprint(), src_hash(suite), C.file_sha(os.path.join(suite.dir, rel)),
           C.file_sha(__file__)]
    want = C.cached_value("lua-official", key, lambda: run_official(R, ref_lua, _tmp(work, "ref"), suite, name, 900))
    if want["timeout"]:
        return R.Result("FAIL", "run", "reference lua timed out; add to skip.txt")
    path, fail = hcc_lua(R, suite, cc_argv)
    if fail:
        return fail.result(R)
    got = run_official(R, path, _tmp(work, "got"), suite, name, RUN_TIMEOUT)
    return C.compare_runs(R, got, want)


def run_test(suite, rel, cc_argv, work, R):
    if rel.startswith("src/"):
        return run_src_test(suite, rel, cc_argv, work, R)
    return run_testes_test(suite, rel, cc_argv, work, R)
