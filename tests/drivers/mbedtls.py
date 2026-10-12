"""Driver for the `mbedtls` suite: Mbed TLS 3.6.2 (release tarball, sha256-pinned) built by hcc.

The default mbedtls_config.h is used.  The upstream unit tests are generated exactly as tests/Makefile does
(framework/scripts/generate_*_tests.py for the generated data, generate_test_code.py for every
tests/suites/test_suite_*.{function,data}) into a private copy of the tree under tests/cache/mbedtls-tree.  Each
test_suite_X is one program: its generated C file, the 22 test helper files and the 108 library files.  The
reference is the same program built entirely by gcc -O0; every test case is compared with the reference's status.

Test ids (tests/suites/mbedtls/manifest.json lists them; regenerate it with tests/scripts/gen_mbedtls_manifest.py):
  mbedtls/case/<suite>/<nnnn>_<slug>   one test case (one block of the suite's .data) of the program built
                       entirely by hcc, except that a unit hcc cannot compile yet is replaced by its gcc
                       object (that unit has its own test below), so a failing case means hcc miscompiled
                       something the case exercises.  Only cases the gcc build runs (not skipped by unmet
                       dependencies) are listed; suites with more than 600 of them are sampled with a fixed
                       stride (recorded in the manifest).
  mbedtls/src/<lib>.c, mbedtls/src/tests/<f>.c   hcc compiles that one library / test-helper file; it is linked
                       with the gcc objects of everything else and the smoke suite md must give the
                       reference's status for every case.
  mbedtls/suite/<suite>   hcc compiles the generated test_suite_<suite>.c; the rest is gcc; every case must match.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading

import common as C
import projlib as P

URL = "https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-3.6.2/mbedtls-3.6.2.tar.bz2"
SHA256 = "8b54fb9bcf4d5a7078028e0520acddefb7900b3e66fec7f7175bb5b7d85ccdca"
SMOKE = ["md"]
RUN_TIMEOUT = 300
RUN_VERSION = "1"  # bump when run_module / run_suite / parse change: cached reference runs are keyed on it
STATUS = re.compile(r"^(.*?)(?: \.+)? (PASS|FAILED|----)$")
MAX_LIVE = 600
# suites left out to keep a full run near ten minutes: slow under an unoptimised compiler (gcc -O0 needs >5 s) or
# thousands of near-identical generated PSA cases
EXCLUDE = {"pkparse", "psa_crypto", "psa_crypto_storage_format.v0", "psa_crypto_storage_format.current",
           "psa_crypto_op_fail.generated", "ssl", "pkcs1_v21"}


_memo_lock = threading.Lock()
_memo = {}


def memo(key, fn):
    """Compute fn() once per process for `key` (thousands of case tests share a few dozen programs and runs)."""
    with _memo_lock:
        e = _memo.setdefault(key, {"lock": threading.Lock(), "done": False, "val": None})
    with e["lock"]:
        if not e["done"]:
            e["val"] = fn()
            e["done"] = True
        return e["val"]


def manifest(suite):
    p = os.path.join(suite.dir, "manifest.json")
    if not os.path.exists(p):
        return {"suites": {}, "lib": [], "helpers": []}
    with open(p) as f:
        return json.load(f)


def slug(desc):
    return re.sub(r"[^A-Za-z0-9]+", "-", desc).strip("-")[:48] or "case"


def list_tests(suite):
    m = manifest(suite)
    out = ["src/%s.c" % u for u in m["lib"]] + ["src/tests/%s.c" % h for h in m["helpers"]]
    out += ["suite/%s" % s for s in sorted(m["suites"])]
    for s, info in sorted(m["suites"].items()):
        out += ["case/%s/%04d_%s" % (s, idx, sl) for idx, sl in info["cases"]]
    return out


# ------------------------------------------------------------------ the tree

def tree():
    """Private generated copy of the release tree (test data + one C file and .datax per suite)."""
    src = P.fetch_tarball("mbedtls", URL, SHA256)
    base = os.path.join(C.cache_root(), "mbedtls-tree", C.sha(SHA256, "v1"))
    t = os.path.join(base, "t")
    with C.locked(base):
        if not os.path.exists(os.path.join(base, "built")):
            shutil.rmtree(t, ignore_errors=True)
            os.makedirs(base, exist_ok=True)
            shutil.copytree(src, t, symlinks=True)
            tests = os.path.join(t, "tests")
            for gen in ("generate_bignum_tests.py", "generate_ecp_tests.py", "generate_config_tests.py", "generate_psa_tests.py"):
                run_py(["python3", os.path.join("..", "framework", "scripts", gen)], tests)
            sdir = os.path.join(tests, "suites")
            for f in sorted(os.listdir(sdir)):
                if f.startswith("test_suite_") and f.endswith(".data"):
                    b = f[:-5]
                    run_py(["python3", "../framework/scripts/generate_test_code.py", "-f", "suites/%s.function" % b.split(".")[0],
                            "-d", "suites/%s" % f, "-t", "suites/main_test.function", "-p", "suites/host_test.function",
                            "-s", "suites", "--helpers-file", "suites/helpers.function", "-o", "."], tests)
            open(os.path.join(base, "built"), "w").close()
    return t


def run_py(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("mbedtls tree: %s failed:\n%s" % (" ".join(cmd), (r.stdout + r.stderr)[-1500:]))


def source_lists(t):
    lib = sorted(f[:-2] for f in os.listdir(os.path.join(t, "library")) if f.endswith(".c"))
    helpers = []
    for dp, _, fn in os.walk(os.path.join(t, "tests", "src")):
        for f in fn:
            if f.endswith(".c"):
                helpers.append(os.path.relpath(os.path.join(dp, f), os.path.join(t, "tests", "src"))[:-2])
    return lib, sorted(helpers)


def project(R, suite, cc_argv, names=None):
    t = tree()
    lib, helpers = source_lists(t)
    units = {"lib/" + u: os.path.join(t, "library", u + ".c") for u in lib}
    units.update({"tests/" + h: os.path.join(t, "tests", "src", h + ".c") for h in helpers})
    for s in (names if names is not None else manifest(suite)["suites"]):
        units["suite/" + s] = os.path.join(t, "tests", "test_suite_%s.c" % s)
    flags = ["-I" + os.path.join(t, "include"), "-I" + os.path.join(t, "library"), "-I" + os.path.join(t, "tests", "include"),
             "-I" + os.path.join(t, "tests", "src"), "-D_FILE_OFFSET_BITS=64", "-D_POSIX_C_SOURCE=200809L"]
    proj = P.Project(R, suite, cc_argv, "mbedtls", units, flags, ["-lm"])
    base = ["lib/" + u for u in lib] + ["tests/" + h for h in helpers]
    return proj, t, base


# ------------------------------------------------------------------ running

def parse(text):
    """Per-case statuses in order, and the first detail line of failures."""
    statuses, details, last = [], {}, None
    for line in text.splitlines():
        m = STATUS.match(line)
        if m:
            statuses.append(m.group(2))
            last = len(statuses) - 1
        elif last is not None and statuses[last] == "FAILED" and last not in details and line.strip():
            details[last] = line.strip()
    return statuses, details


def run_suite(R, t, exe, name, work):
    d = P.tmpdir(work, "run-" + name.replace(".", "_"))
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    link = os.path.join(d, "framework")
    if not os.path.lexists(link):
        os.symlink(os.path.join(t, "framework"), link)
    rc, out, err, to = P.run_cmd(R, [exe, os.path.join(t, "tests", "test_suite_%s.datax" % name)], RUN_TIMEOUT, cwd=os.path.join(d, "tests"))
    text = out[:4000000].decode("utf-8", "replace")
    statuses, details = parse(text)
    return {"rc": rc, "timeout": to, "statuses": statuses, "descs": [m.group(1) for m in map(STATUS.match, text.splitlines()) if m],
            "details": {str(k): v for k, v in details.items()}, "tail": text[-300:] + R.text(err)[-300:]}


def cached_run(R, t, exe, name, work, who):
    key = C.sha(who, C.file_sha(exe), name, RUN_VERSION)
    p = os.path.join(C.cache_root(), "val", "mbedtls-run", key[:2], key + ".json")
    with C.locked(p):
        v = C.read_json(p)
        if v is not None:
            return v["v"]
        v = run_suite(R, t, exe, name, work)
        if not v["timeout"]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            C.atomic_write(p, json.dumps({"v": v}))
        return v


def units_of(base, name):
    return base + ["suite/" + name]


def compare_case(R, got, want, idx):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "suite run >%ds" % RUN_TIMEOUT)
    st = got["statuses"]
    if idx < len(st):
        if st[idx] == want["statuses"][idx]:
            return R.Result("PASS")
        why = got["details"].get(str(idx), "")
        return R.Result("FAIL", "compare", R.short("case %s, expected %s %s" % (st[idx], want["statuses"][idx], why)))
    if got["rc"] < 0:
        return R.Result("FAIL", "run", R.short("suite killed by %s after %d of %d cases %s" % (R.signame(got["rc"]), len(st), len(want["statuses"]), got["tail"][-100:])))
    return R.Result("FAIL", "run", R.short("only %d of %d cases ran (exit %d) %s" % (len(st), len(want["statuses"]), got["rc"], got["tail"][-100:])))


def compare_suite(R, got, want, what):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "suite run >%ds" % RUN_TIMEOUT)
    for i, w in enumerate(want["statuses"]):
        if i >= len(got["statuses"]):
            return R.Result("FAIL", "run", R.short("%s: only %d of %d cases ran (exit %d)" % (what, len(got["statuses"]), len(want["statuses"]), got["rc"])))
        if got["statuses"][i] != w:
            return R.Result("FAIL", "compare", R.short("%s: case %d %s, expected %s %s" % (what, i, got["statuses"][i], w, got["details"].get(str(i), ""))))
    return R.Result("PASS")


def reference(R, proj, t, base, name, work):
    ref = proj.ref("suite-" + name, units_of(base, name), ["-lm"])
    return cached_run(R, t, ref, name, work, "ref")


def run_test(suite, rel, cc_argv, work, R):
    cc = tuple(cc_argv)
    proj, t, base = memo(("proj", cc), lambda: project(R, suite, cc_argv))
    kind, _, rest = rel.partition("/")
    if kind == "case":
        name, _, cs = rest.partition("/")
        idx = int(cs.split("_", 1)[0])
        exe, fail = memo(("prog", cc, name), lambda: proj.full("suite-" + name, units_of(base, name), ["-lm"], degrade=True))
        if fail:
            return fail.result(R)
        want = memo(("ref", name), lambda: reference(R, proj, t, base, name, work))
        got = memo(("run", cc, name), lambda: cached_run(R, t, exe, name, work, "hcc"))
        return compare_case(R, got, want, idx)
    if kind == "suite":
        unit, smoke = "suite/" + rest, [rest]
    else:
        unit = ("tests/" + rest[len("tests/"):-2]) if rest.startswith("tests/") else "lib/" + rest[:-2]
        smoke = SMOKE
    for name in smoke:
        out = os.path.join(work, "exe-" + name)
        f = proj.mixed(out, units_of(base, name), [unit], ["-lm"])
        if f:
            return f.result(R)
        want = memo(("ref", name), lambda: reference(R, proj, t, base, name, work))
        res = compare_suite(R, run_suite(R, t, out, name, P.tmpdir(work, "got")), want, "test_suite_" + name)
        if res.status != "PASS":
            return res
    return R.Result("PASS")


# ------------------------------------------------------- manifest generation

def build_manifest(R, suite):
    """Used by tests/scripts/gen_mbedtls_manifest.py: build every suite with gcc, run it, list the live cases."""
    import tempfile
    work = tempfile.mkdtemp(prefix="mbed-manifest-")
    t = tree()
    sdir = os.path.join(t, "tests", "suites")
    names = sorted(f[len("test_suite_"):-5] for f in os.listdir(sdir) if f.startswith("test_suite_") and f.endswith(".data"))
    proj, t, base = project(R, suite, ["gcc"], names)
    lib, helpers = source_lists(t)
    out = {"lib": lib, "helpers": helpers, "suites": {}}
    for n in names:
        if n in EXCLUDE:
            continue
        want = reference(R, proj, t, base, n, work)
        if want["timeout"] or want["rc"] != 0 or "FAILED" in want["statuses"]:
            print("skipping suite %s: reference run rc=%s timeout=%s failed=%d" % (n, want["rc"], want["timeout"], want["statuses"].count("FAILED")))
            continue
        live = [(i, slug(want["descs"][i])) for i, s in enumerate(want["statuses"]) if s == "PASS"]
        if not live:
            continue
        stride = -(-len(live) // MAX_LIVE)
        out["suites"][n] = {"total": len(want["statuses"]), "live": len(live), "stride": stride, "cases": live[::stride]}
    shutil.rmtree(work, ignore_errors=True)
    return out
