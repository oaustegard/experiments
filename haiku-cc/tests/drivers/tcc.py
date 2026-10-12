"""Driver for the `tcc` suite: TinyCC (mob branch, commit-pinned) built by hcc.

The reference is a gcc-built tcc from the upstream `./configure && make` (in a private copy of the tree
under tests/cache/tcc-tree, which also supplies libtcc1.a, the runtime objects, config.h and tccdefs_.h).
The compiler under test builds the 11 translation units of the native x86_64 compiler (ONE_SOURCE=0).
Every test runs the same command line with the hcc-built tcc and with the reference tcc and compares what
they print (stdout+stderr, paths normalised), their exit status and the files they produce.

Test ids:
  tcc/src/<u>.c        hcc compiles that one unit; linked with the gcc -O0 objects of the other ten into a
                       tcc that must pass a smoke set: 12 tests2 programs, tcctest (-run), hello (exe).
  tcc/t2/<name>.c      upstream tests/tests2/<name>.c run by an entirely hcc-built tcc exactly as
                       tests/tests2/Makefile does (-run, per-test flags/arguments, -dt, -b, ...).
  tcc/pp/<name>        upstream tests/pp (tcc -E -P).
  tcc/obj/<name>.c     tcc -c on a tests2 file: the object file must be byte-identical to the reference's.
  tcc/self/<name>      upstream tests/Makefile targets: test1 (tcc -run tcctest.c), test3 (tcc compiled by
                       tcc compiled by tcc runs tcctest.c), tcctest as object/exe, hello-exe/run, vla_test,
                       asm-c-connect, cross-<target> (tcc.c run as a cross compiler of tcctest.c), exe-tcc
                       (tcc.c compiled to an executable, byte-identical to the reference's), obj-<unit>
                       (every tcc source unit compiled to an object, byte-identical).
  tcc/lib/abitest-cc, tcc/lib/abitest-tcc, tcc/lib/libtest   libtcc built by hcc used from C programs.
"""
import hashlib
import os
import re
import shlex
import shutil
import subprocess

import common as C
import projlib as P

URL = "https://github.com/TinyCC/tinycc.git"
REF = "mob"
COMMIT = "43c7708b85681a2fd4451c8a541af4494a8919b2"
UNITS = ["tcc", "libtcc", "tccpp", "tccgen", "tccdbg", "tccelf", "tccasm", "tccrun", "x86_64-gen", "x86_64-link", "i386-asm"]
LIBUNITS = [u for u in UNITS if u != "tcc"]
LDLIBS = ["-lm", "-ldl"]
T2_SKIP = {"34_array_assignment", "98_al_ax_extend", "99_fastcall", "138_arm64_encoding", "139_arm64_errors",
           "140_arm64_extasm", "141_riscv_asm", "145_winarm64_interlocked"}  # tests2/Makefile SKIP for x86_64 linux
SMOKE = ["00_assignment", "04_for", "07_function", "18_include", "22_floating_point", "28_strings", "30_hanoi",
         "46_grep", "55_lshift_type", "70_floating_point_literals", "95_bitfields", "101_cleanup"]
SELF = ["test1", "test3", "obj-tcctest", "exe-tcctest", "hello-exe", "hello-run", "vla_test", "asm-c-connect", "exe-tcc"]
CROSS = {
    "i386": "-DTCC_TARGET_I386", "i386-win32": "-DTCC_TARGET_I386 -DTCC_TARGET_PE",
    "x86_64": "-DTCC_TARGET_X86_64", "x86_64-win32": "-DTCC_TARGET_X86_64 -DTCC_TARGET_PE",
    "x86_64-osx": "-DTCC_TARGET_X86_64 -DTCC_TARGET_MACHO",
    "arm-eabihf": "-DTCC_TARGET_ARM -DTCC_ARM_VFP -DTCC_ARM_EABI -DTCC_ARM_HARDFLOAT",
    "arm-fpa": "-DTCC_TARGET_ARM", "arm64": "-DTCC_TARGET_ARM64", "arm64-osx": "-DTCC_TARGET_ARM64 -DTCC_TARGET_MACHO",
    "riscv64": "-DTCC_TARGET_RISCV64", "c67": "-DTCC_TARGET_C67 -w",
}
HEXFILTER = ("112_backtrace", "113_btdll", "126_bound_global")
# per-test command lines of tests/tests2/Makefile: {TCC} {F} {N} {SRC} are substituted
SPEC = {
    "31_args": {"args": "arg1 arg2 arg3 arg4 arg5"},
    "46_grep": {"args": "'[^* ]*[:a:d: ]+\\:\\*-/: $' {SRC}/46_grep.c"},
    "76_dollars_in_identifiers": {"flags": "-fdollars-in-identifiers"},
    "22_floating_point": {"flags": "-lm"}, "24_math_library": {"flags": "-lm"},
    "60_errors_and_warnings": {"flags": "-dt"}, "96_nodata_wanted": {"flags": "-dt"},
    "125_atomic_misc": {"flags": "-dt"}, "128_run_atexit": {"flags": "-dt"},
    "42_function_pointer": {"norun": True},
    "104_inline": {"flags": "{SRC}/104+_inline.c"},
    "120_alias": {"flags": "{SRC}/120+_alias.c", "norun": True},
    "106_versym": {"flags": "-pthread", "norun": True},
    "108_constructor": {"norun": True},
    "112_backtrace": {"flags": "-dt -b"},
    "113_btdll": {"script": "{TCC} -bt {F} -shared -D DLL=1 -o a1.so && {TCC} -bt {F} -shared -D DLL=2 -o a2.so && "
                            "{TCC} -bt {F} a1.so a2.so -Wl,-rpath=. -o {N}.exe && ./{N}.exe"},
    "114_bound_signal": {"flags": "-b", "norun": True}, "115_bound_setjmp": {"flags": "-b"},
    "116_bound_setjmp2": {"flags": "-b"},
    "117_builtins": {"script": "( {TCC} -run {F} && {TCC} -b -run {F} )"},
    "121_struct_return": {"flags": "-b"}, "122_vla_reuse": {"flags": "-b"},
    "124_atomic_counter": {"flags": "-pthread"},
    "126_bound_global": {"flags": "-b", "norun": True}, "132_bound_test": {"flags": "-b"},
    "144_tls": {"flags": "-pthread", "norun": True},
    "146_tls_extern": {"script": "{TCC} -c {F} -o {N}-main.o && {TCC} -c {F} -DDEFS -o {N}-defs.o && "
                                 "gcc -no-pie -Wl,-z,noexecstack {N}-main.o {N}-defs.o -o {N}.exe && ./{N}.exe"},
    "148_linker_symbols": {"script": "for T in 1 2 3 4 5 6; do echo; echo \"--- TEST $T ---\"; F={N}-$T; "
                                     "{TCC} {F} -DTEST=$T+100 -c -fcommon -o $F.o && {TCC} {F} -DTEST=$T+200 -shared -o $F.so -fPIC && "
                                     "{TCC} {F} -DTEST=$T -Wl,-rpath=. -o $F.exe $F.o $F.so -no-pie && ./$F.exe; done"},
}
TIMEOUT = 60


# ------------------------------------------------------------------ tree

def src_dir():
    return P.fetch_git("tinycc", URL, REF, COMMIT)


def tree():
    """A configured and gcc-built copy of the tree (reference tcc, libtcc1.a, runtime objects, config.h)."""
    src = src_dir()
    base = os.path.join(C.cache_root(), "tcc-tree", C.sha(COMMIT, "v1"))
    t = os.path.join(base, "t")
    with C.locked(base):
        if not os.path.exists(os.path.join(base, "built")):
            shutil.rmtree(t, ignore_errors=True)
            os.makedirs(base, exist_ok=True)
            shutil.copytree(src, t, ignore=shutil.ignore_patterns(".git"), symlinks=True)
            for cmd in (["./configure", "--cc=gcc", "--prefix=/nonexistent/tcc"], ["make", "-j1"]):
                r = subprocess.run(cmd, cwd=t, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError("tcc reference tree: %s failed:\n%s" % (" ".join(cmd), (r.stdout + r.stderr)[-2000:]))
            open(os.path.join(base, "built"), "w").close()
    return t


def t2_names():
    d = os.path.join(src_dir(), "tests", "tests2")
    return sorted(f[:-2] for f in os.listdir(d) if re.match(r"^\d{2,3}_.*\.c$", f) and f[:-2] not in T2_SKIP)


def pp_names():
    d = os.path.join(src_dir(), "tests", "pp")
    return sorted(f for f in os.listdir(d) if f.endswith((".c", ".S")))


def list_tests(suite):
    names = t2_names()
    out = ["src/%s.c" % u for u in UNITS]
    out += ["t2/%s.c" % n for n in names] + ["obj/%s.c" % n for n in names]
    out += ["pp/%s" % n for n in pp_names()]
    out += ["self/%s" % n for n in SELF + ["cross-" + c for c in CROSS] + ["obj-" + u for u in UNITS]]
    out += ["lib/abitest-cc", "lib/abitest-tcc", "lib/libtest"]
    return out


def project(R, suite, cc_argv):
    t = tree()
    units = {u: os.path.join(t, u + ".c") for u in UNITS}
    units["abitest"] = os.path.join(t, "tests", "abitest.c")
    units["libtcc_test"] = os.path.join(t, "tests", "libtcc_test.c")
    unit_flags = {"tcc": ["-DONE_SOURCE=0"], "libtcc": ["-DONE_SOURCE=0"], "abitest": ["-DONE_SOURCE=0"],
                  "libtcc_test": ["-DONE_SOURCE=0"]}
    return P.Project(R, suite, cc_argv, "tcc", units, ["-I" + t], LDLIBS, unit_flags), t


def base_args(t):
    return ["-B" + t, "-I" + os.path.join(t, "include"), "-I" + t]


# --------------------------------------------------------- running scenarios

def norm(s, t, cwd, hexfilter=False):
    s = s.replace(t + "/tests/tests2/", "").replace(t + "/tests/pp/", "").replace(t + "/tests/", "").replace(t, "<TREE>")
    s = s.replace(cwd, "<CWD>")
    if hexfilter:
        s = re.sub(r"[0-9A-Fa-fx]{5,}", "........", s)
        s = re.sub(r"0x[0-9A-Fa-f]{1,}", "0x?", s)
    return s


def shell(R, script, cwd, t, timeout=TIMEOUT, hexfilter=False):
    rc, out, err, to = P.run_cmd(R, ["sh", "-c", "( %s ) 2>&1" % script], timeout, cwd=cwd)
    files = {}
    for fn in sorted(os.listdir(cwd)):
        p = os.path.join(cwd, fn)
        if os.path.isfile(p) and not fn.endswith((".c", ".h", ".S")):
            files[fn] = C.file_sha(p) if not fn.endswith((".exe", ".so")) or fn.startswith("obj") else "present"
    return {"rc": rc, "out": norm(R.text(out), t, cwd, hexfilter), "files": files, "timeout": to}


def sh_quote(args):
    return " ".join(shlex.quote(a) for a in args)


def scen_t2(R, t, exe, name, cwd):
    src = os.path.join(t, "tests", "tests2")
    spec = SPEC.get(name, {})
    sub = {"TCC": sh_quote([exe] + base_args(t)), "F": os.path.join(src, name + ".c"), "N": name, "SRC": src}

    def fmt(s):
        for k, v in sub.items():
            s = s.replace("{%s}" % k, v)
        return s
    if "script" in spec:
        script = fmt(spec["script"])
    else:
        flags, args = fmt(spec.get("flags", "")), fmt(spec.get("args", ""))
        if spec.get("norun"):
            script = "%s %s %s -o %s.exe && ./%s.exe %s" % (sub["TCC"], flags, sub["F"], name, name, args)
        else:
            script = "%s %s -run %s %s" % (sub["TCC"], flags, sub["F"], args)
    return shell(R, script, cwd, t, hexfilter=name in HEXFILTER)


def scen_obj(R, t, exe, name, cwd):
    src = os.path.join(t, "tests", "tests2")
    for f in os.listdir(src):
        if f.endswith((".c", ".h")):
            shutil.copy(os.path.join(src, f), cwd)
    r = shell(R, "%s -c %s.c -o obj-%s.o" % (sh_quote([exe] + base_args(t)), name, name), cwd, t)
    return r


def scen_pp(R, t, exe, name, cwd):
    f = os.path.join(t, "tests", "pp", name)
    return shell(R, "%s -E -P %s" % (sh_quote([exe] + base_args(t)), shlex.quote(f)), cwd, t)


def scen_self(R, t, exe, name, cwd):
    TCC = sh_quote([exe] + base_args(t))
    tests = os.path.join(t, "tests")
    flags = sh_quote(base_args(t))
    if name == "test1":
        return shell(R, "%s -w -run %s/tcctest.c" % (TCC, tests), cwd, t, 120)
    if name == "test3":
        run = "-run %s/tcc.c %s" % (t, flags)
        return shell(R, "%s %s %s %s -w -run %s/tcctest.c" % (TCC, run, run, run, tests), cwd, t, 600)
    if name == "obj-tcctest":
        return shell(R, "%s -c -o tcctest3.o %s/tcctest.c && %s -o tcctest3 tcctest3.o && ./tcctest3" % (TCC, tests, TCC), cwd, t, 120)
    if name == "exe-tcctest":
        return shell(R, "%s -o tcctest1 %s/tcctest.c && ./tcctest1" % (TCC, tests), cwd, t, 120)
    if name == "hello-exe":
        return shell(R, "%s %s/examples/ex1.c -o hello && ./hello" % (TCC, t), cwd, t)
    if name == "hello-run":
        return shell(R, "%s -run %s/examples/ex1.c" % (TCC, t), cwd, t)
    if name == "vla_test":
        return shell(R, "%s -o vla_test %s/vla_test.c && ./vla_test" % (TCC, tests), cwd, t)
    if name == "asm-c-connect":
        return shell(R, "%s -o a1 %s/asm-c-connect-1.c %s/asm-c-connect-2.c && ./a1 && "
                        "%s -c -o c1.o %s/asm-c-connect-1.c && %s -c -o c2.o %s/asm-c-connect-2.c && %s -o a2 c1.o c2.o && ./a2"
                     % (TCC, tests, tests, TCC, tests, TCC, tests, TCC), cwd, t)
    if name == "exe-tcc":
        return shell(R, "%s %s/tcc.c %s -o obj-tcc2.exe && ./obj-tcc2.exe %s %s/examples/ex1.c -o hello && ./hello"
                     % (TCC, t, flags, flags, t), cwd, t, 300)
    if name.startswith("cross-"):
        d = CROSS[name[6:]]
        return shell(R, "%s %s -DTCC_CROSS_TEST -run %s/tcc.c -c %s/tcctest.c -w %s" % (TCC, d, t, tests, flags), cwd, t, 300)
    if name.startswith("obj-"):
        u = name[4:]
        extra = "-DONE_SOURCE=0" if u in ("tcc", "libtcc") else ""
        return shell(R, "%s %s -c %s/%s.c -o obj-%s.o" % (TCC, extra, t, u, u), cwd, t, 120)
    raise KeyError(name)


def scenario(R, t, rel, exe, cwd):
    kind, _, name = rel.partition("/")
    if kind == "t2":
        return scen_t2(R, t, exe, name[:-2], cwd)
    if kind == "obj":
        return scen_obj(R, t, exe, name[:-2], cwd)
    if kind == "pp":
        return scen_pp(R, t, exe, name, cwd)
    return scen_self(R, t, exe, name, cwd)


def compare(R, got, want, what=""):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "run >%ds" % TIMEOUT)
    if want.get("flaky"):
        want = dict(want, out=got["out"], files=got["files"])
    if got["rc"] < 0:
        return R.Result("FAIL", "run", R.short("killed by %s %s" % (R.signame(got["rc"]), got["out"][-200:])))
    if got["out"] != want["out"]:
        return R.Result("FAIL", "compare", R.short(R.describe_diff(want["out"].encode(), got["out"].encode())), got["out"][-3000:])
    if got["rc"] != want["rc"]:
        return R.Result("FAIL", "compare", R.short("exit %d, expected %d: %s" % (got["rc"], want["rc"], got["out"][-120:])))
    if got["files"] != want["files"]:
        bad = sorted(k for k in set(got["files"]) | set(want["files"]) if got["files"].get(k) != want["files"].get(k))
        return R.Result("FAIL", "compare", R.short("files differ from reference: %s" % ", ".join(bad)))
    return R.Result("PASS")


def reference(R, t, rel, work):
    ref = os.path.join(t, "tcc")
    mk = [0]

    def once():
        mk[0] += 1
        return scenario(R, t, rel, ref, P.tmpdir(work, "ref%d-%s" % (mk[0], rel.replace("/", "_"))))

    def compute():
        a, b = once(), once()
        if a["out"] != b["out"] or a["files"] != b["files"] or a["rc"] != b["rc"]:
            a["flaky"] = True
        return a
    return P.ref_value("tcc-ref", [C.file_sha(ref), rel, C.file_sha(__file__)], compute)


def run_test(suite, rel, cc_argv, work, R):
    proj, t = project(R, suite, cc_argv)
    kind, _, name = rel.partition("/")
    if kind == "src":
        return run_src(R, proj, t, name[:-2], work)
    if kind == "lib":
        return run_lib(R, proj, t, name, work)
    exe, fail = proj.full("tcc", UNITS, LDLIBS)
    if fail:
        return fail.result(R)
    want = reference(R, t, rel, work)
    got = scenario(R, t, rel, exe, P.tmpdir(work, "got"))
    return compare(R, got, want)


def run_src(R, proj, t, unit, work):
    exe = os.path.join(work, "tcc-mixed")
    f = proj.mixed(exe, UNITS, [unit], LDLIBS)
    if f:
        return f.result(R)
    rels = ["t2/%s.c" % n for n in SMOKE] + ["self/hello-exe", "self/obj-tcctest"]
    for rel in rels:
        res = compare(R, scenario(R, t, rel, exe, P.tmpdir(work, "got-" + rel.replace("/", "_"))), reference(R, t, rel, work))
        if res.status != "PASS":
            res.reason = R.short("%s: %s" % (rel, res.reason))
            return res
    return R.Result("PASS")


def lib_scenario(R, t, exe, name, cwd):
    flags = base_args(t)
    if name == "abitest-cc" or name == "libtest":
        return shell(R, "%s %s" % (shlex.quote(exe), sh_quote(flags if name == "abitest-cc" else [os.path.join(t, "tcc.c")] + flags)), cwd, t)
    raise KeyError(name)


def run_lib(R, proj, t, name, work):
    if name == "abitest-tcc":
        # the reference tcc and the hcc-built tcc both compile abitest.c + libtcc.c into an executable and run it
        exe, fail = proj.full("tcc", UNITS, LDLIBS)
        if fail:
            return fail.result(R)

        def scen(tcc, cwd):
            args = sh_quote([tcc] + base_args(t))
            r = shell(R, "%s -o abitest-tcc.exe %s/tests/abitest.c %s/libtcc.c -ldl -lm && ./abitest-tcc.exe %s"
                      % (args, t, t, sh_quote(base_args(t))), cwd, t, 120)
            return r
        want = P.ref_value("tcc-ref", [C.file_sha(os.path.join(t, "tcc")), "abitest-tcc", C.file_sha(__file__)],
                           lambda: scen(os.path.join(t, "tcc"), P.tmpdir(work, "ref")))
        return compare(R, scen(exe, P.tmpdir(work, "got")), want)
    prog = "abitest" if name == "abitest-cc" else "libtcc_test"
    units = [prog] + LIBUNITS
    exe = os.path.join(work, prog)
    f = proj.mixed(exe, units, LIBUNITS, LDLIBS)
    if f:
        return f.result(R)
    ref = proj.ref("lib-" + prog, units, LDLIBS)
    want = P.ref_value("tcc-ref", [C.file_sha(ref), name, C.file_sha(__file__)], lambda: lib_scenario(R, t, ref, name, P.tmpdir(work, "ref")))
    return compare(R, lib_scenario(R, t, exe, name, P.tmpdir(work, "got")), want)
