"""Driver for the `chibicc-self` suite: chibicc (rui314/chibicc, commit-pinned) built by hcc, then used
as a compiler: hcc-compiled code compiling C.  The reference is the same tree built by gcc -O0.

Test ids:
  chibicc-self/src/<u>.c      hcc compiles that one translation unit (9 in the tree); linked with the gcc
                              objects of the others.  The resulting chibicc must emit byte-identical assembly
                              for 6 of chibicc's own test files and build + run test/arith.c.
  chibicc-self/test/<n>.c     upstream `make test`: the all-hcc chibicc compiles test/<n>.c (-Iinclude -Itest),
                              the object is linked with gcc against test/common and must exit 0 with the same
                              output as the gcc-built chibicc's; its -S output must be byte-identical.
  chibicc-self/driver         upstream test/driver.sh run against the all-hcc chibicc.
  chibicc-self/stage2/<u>.c   the all-hcc chibicc compiles chibicc's own source <u>.c (-S): the assembly must be
                              byte-identical to what the gcc-built chibicc produces.
  chibicc-self/stage2-test/<n>.c  upstream `make test-stage2`: stage2 chibicc = chibicc's sources compiled by
                              the hcc-built chibicc and linked by gcc; it compiles test/<n>.c like above.
"""
import hashlib
import os
import shutil

import common as C
import projlib as P

URL = "https://github.com/rui314/chibicc.git"
REF = "main"
COMMIT = "90d1f7f199cc55b13c7fdb5839d1409806633fdb"
UNITS = ["codegen", "hashmap", "main", "parse", "preprocess", "strings", "tokenize", "type", "unicode"]
SMOKE = ["arith", "control", "function", "struct", "float", "varargs"]
FLAGS = ["-std=c11", "-fno-common"]
TIMEOUT = 60


def root():
    return P.fetch_git("chibicc", URL, REF, COMMIT)


def test_names():
    d = os.path.join(root(), "test")
    return sorted(f[:-2] for f in os.listdir(d) if f.endswith(".c"))


def list_tests(suite):
    names = test_names()
    return (["src/%s.c" % u for u in UNITS] + ["test/%s.c" % n for n in names] + ["driver"]
            + ["stage2/%s.c" % u for u in UNITS] + ["stage2-test/%s.c" % n for n in names])


def project(R, suite, cc_argv):
    r = root()
    units = {u: os.path.join(r, u + ".c") for u in UNITS}
    return P.Project(R, suite, cc_argv, "chibicc", units, FLAGS), r


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rundir(r, exe, d):
    """Directory laid out like the chibicc checkout: ./chibicc, ./include, ./test (chibicc finds its headers
    relative to argv[0])."""
    os.makedirs(os.path.join(d, "out"), exist_ok=True)
    for name, target in (("chibicc", exe), ("include", os.path.join(r, "include")), ("test", os.path.join(r, "test"))):
        link = os.path.join(d, name)
        if not os.path.lexists(link):
            os.symlink(target, link)
    return d


def runb(R, argv, cwd, timeout=TIMEOUT):
    rc, out, err, to = P.run_cmd(R, argv, timeout, cwd=cwd)
    return {"rc": rc, "out": R.text(out), "err": R.text(err), "timeout": to}


def asm_of(R, d, source, name):
    """chibicc -S: returns {rc, asm sha, err}."""
    out = os.path.join(d, "out", name + ".s")
    r = runb(R, ["./chibicc", "-Iinclude", "-Itest", "-S", "-o", out, source], d)
    asm = open(out, "rb").read() if os.path.exists(out) else b""
    src = open(os.path.join(d, source) if not os.path.isabs(source) else source, errors="replace").read()
    clock = any(m in src for m in ("__TIME__", "__DATE__", "__TIMESTAMP__"))  # the assembly embeds the build time
    return {"rc": r["rc"], "asm": "clock" if clock else sha(asm), "err": r["err"][-300:], "timeout": r["timeout"]}


def build_run_test(R, r, d, name):
    """Upstream make test: chibicc -c, gcc link with test/common, run."""
    obj = os.path.join(d, "out", name + ".o")
    c = runb(R, ["./chibicc", "-Iinclude", "-Itest", "-c", "-o", obj, "test/%s.c" % name], d)
    if c["rc"] != 0 or c["timeout"]:
        return {"stage": "compile", "rc": c["rc"], "err": c["err"][-300:], "timeout": c["timeout"], "out": ""}
    exe = os.path.join(d, "out", name + ".exe")
    l = runb(R, ["gcc", "-pthread", "-o", exe, obj, "-xc", os.path.join(r, "test", "common")], d)
    if l["rc"] != 0:
        return {"stage": "link", "rc": l["rc"], "err": l["err"][-300:], "timeout": False, "out": ""}
    x = runb(R, [exe], d)
    src = open(os.path.join(r, "test", name + ".c"), errors="replace").read()
    clock = any(m in src for m in ("__TIME__", "__DATE__", "__TIMESTAMP__"))  # the program prints the build time
    return {"stage": "run", "rc": x["rc"], "err": x["err"][-300:], "timeout": x["timeout"],
            "out": "clock" if clock else sha(x["out"].encode())}


def compare_asm(R, got, want, what):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "chibicc >%ds" % TIMEOUT)
    if got["rc"] != want["rc"]:
        return R.Result("FAIL", "compare", R.short("%s: chibicc exit %d, expected %d: %s" % (what, got["rc"], want["rc"], got["err"])))
    if got["asm"] != want["asm"]:
        return R.Result("FAIL", "compare", "%s: assembly differs from the gcc-built chibicc's" % what)
    return R.Result("PASS")


def compare_run(R, got, want, what):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "%s >%ds" % (got["stage"], TIMEOUT))
    if (got["stage"], got["rc"]) != (want["stage"], want["rc"]) or got["out"] != want["out"]:
        stage = got["stage"] if got["stage"] != "run" else ("run" if got["rc"] < 0 else "compare")
        return R.Result("FAIL", {"compile": "codegen"}.get(stage, stage),
                        R.short("%s: %s exit %d (reference: %s exit %d) %s" % (what, got["stage"], got["rc"], want["stage"], want["rc"], got["err"])))
    return R.Result("PASS")


def ref_exe(proj):
    return proj.ref("chibicc", UNITS)


def ref_val(R, proj, r, key, compute):
    return P.ref_value("chibicc-ref", [C.file_sha(ref_exe(proj)), key, C.file_sha(__file__)], compute)


def run_test(suite, rel, cc_argv, work, R):
    proj, r = project(R, suite, cc_argv)
    kind, _, name = rel.partition("/")
    if kind == "src":
        return run_src(R, proj, r, name[:-2], work)
    exe, fail = proj.full("chibicc", UNITS)
    if fail:
        return fail.result(R)
    ref = ref_exe(proj)
    if kind == "test":
        return run_upstream(R, proj, r, exe, ref, name[:-2], work, "t")
    if kind == "driver":
        return run_driver(R, proj, r, exe, ref, work)
    if kind == "stage2":
        u = name[:-2]
        src = os.path.join(r, u + ".c")
        want = ref_val(R, proj, r, "stage2/" + u, lambda: asm_of(R, rundir(r, ref, P.tmpdir(work, "ref")), src, u))
        return compare_asm(R, asm_of(R, rundir(r, exe, P.tmpdir(work, "got")), src, u), want, u + ".c")
    s2, fail = stage2(R, proj, r, exe, work)
    if fail:
        return fail.result(R)
    return run_upstream(R, proj, r, s2, ref, name[:-2], work, "s2")


def run_upstream(R, proj, r, exe, ref, n, work, tag):
    src = "test/%s.c" % n
    d = rundir(r, exe, P.tmpdir(work, "got"))
    want_asm = ref_val(R, proj, r, "asm/" + n, lambda: asm_of(R, rundir(r, ref, P.tmpdir(work, "refa")), src, n))
    res = compare_asm(R, asm_of(R, d, src, n), want_asm, n + ".c")
    if res.status != "PASS":
        return res
    want = ref_val(R, proj, r, "run/" + n, lambda: build_run_test(R, r, rundir(r, ref, P.tmpdir(work, "refb")), n))
    return compare_run(R, build_run_test(R, r, d, n), want, n + ".c")


def stage2(R, proj, r, exe, work):
    def build(out):
        d = rundir(r, exe, P.tmpdir(os.path.dirname(out), "s2"))
        objs = []
        for u in UNITS:
            o = os.path.join(d, "out", u + ".o")
            c = runb(R, ["./chibicc", "-c", "-o", o, os.path.join(r, u + ".c")], d, 300)
            if c["rc"] != 0 or c["timeout"]:
                return C.Failure("FAIL" if not c["timeout"] else "TIMEOUT", "codegen", R.short("stage2 %s.c: %s" % (u, c["err"] or "exit %d" % c["rc"])), c["err"])
            objs.append(o)
        return C.link(R, out, objs, [], d)
    return C.cached_program(R, "chibicc-stage2-hcc", [C.file_sha(exe), " ".join(UNITS), C.file_sha(__file__)], build)


def run_src(R, proj, r, unit, work):
    exe = os.path.join(work, "chibicc-mixed")
    f = proj.mixed(exe, UNITS, [unit])
    if f:
        return f.result(R)
    ref = ref_exe(proj)
    d = rundir(r, exe, P.tmpdir(work, "got"))
    for n in SMOKE:
        src = "test/%s.c" % n
        want = ref_val(R, proj, r, "asm/" + n, lambda: asm_of(R, rundir(r, ref, P.tmpdir(work, "refa-" + n)), src, n))
        res = compare_asm(R, asm_of(R, d, src, n), want, n + ".c")
        if res.status != "PASS":
            return res
    want = ref_val(R, proj, r, "run/arith", lambda: build_run_test(R, r, rundir(r, ref, P.tmpdir(work, "refb")), "arith"))
    return compare_run(R, build_run_test(R, r, d, "arith"), want, "arith.c")


def driver_run(R, d):
    x = runb(R, ["bash", "test/driver.sh", "./chibicc"], d, 120)
    return {"rc": x["rc"], "out": x["out"].replace(d, "<D>"), "err": x["err"][-300:], "timeout": x["timeout"]}


def run_driver(R, proj, r, exe, ref, work):
    want = ref_val(R, proj, r, "driver", lambda: driver_run(R, rundir(r, ref, P.tmpdir(work, "ref"))))
    got = driver_run(R, rundir(r, exe, P.tmpdir(work, "got")))
    return C.compare_runs(R, got, want)
