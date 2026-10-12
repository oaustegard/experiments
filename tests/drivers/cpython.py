"""Driver for the `cpython` suite: CPython 3.12.7 (git tag, commit-pinned) built by hcc.

The tree is configured and built once by gcc (-O0) in a private copy under tests/cache/cpython-tree, with a CC
wrapper that records every compile command; that gives the reference `python`, the generated sources (frozen
modules, deepfreeze.c, pyconfig.h), the shared extension modules (always gcc-built) and, for each of the 156
objects that make up the python executable (Programs/python.o plus the 155 in libpython), the exact flags.
hcc compiles those objects; the executable is linked by gcc with -Wl,--export-dynamic so that the gcc-built
extension modules load into it.  Tests are modules of the standard regression suite run with
`python -m test -v <module>` (PYTHONHASHSEED=0); a test case passes when its unittest line has the status the
gcc-built python gives it.

Test ids (tests/suites/cpython/manifest.json; regenerate with tests/scripts/gen_cpython_manifest.py):
  cpython/case/<module>/<nnnn>_<slug>   one unittest test method of a standard-library test module, run by a
                       python whose objects are all built by hcc, except that an object hcc cannot compile yet is
                       replaced by its gcc object (that object has its own test below).  Only methods that pass
                       with the gcc build in two runs are listed; modules with more than 400 are sampled by stride.
  cpython/src/<dir>/<file>.c   hcc compiles that one translation unit; the other 155 objects are gcc -O0; the
                       resulting python must give the gcc build's status for every case of the smoke modules
                       test_grammar and test_list.
"""
import json
import os
import re
import shutil
import subprocess
import threading

import common as C
import projlib as P

URL = "https://github.com/python/cpython.git"
REF = "v3.12.7"
COMMIT = "0b05ead877f909b7efe712db758012d9dbece7ce"
SMOKE = ["test_grammar", "test_list"]
RUN_TIMEOUT = 300
MAX_LIVE = 400
LDFLAGS = ["-Wl,--export-dynamic", "-ldl", "-lm", "-lpthread"]
LINE = re.compile(r"^(\S+) \(([\w.]+)\)(?:\n[^\n]*?)? \.\.\. (ok|FAIL|ERROR|skipped .*|expected failure|unexpected success)$", re.M)
CANDIDATES = """test_opcodes test_grammar test_syntax test_class test_types test_descr test_scope test_generators test_genexps
test_with test_coroutines test_list test_dict test_unicode_identifiers test_int test_float test_complex test_cmath
test_fractions test_functools test_operator test_bisect test_sort test_enumerate test_slice test_string test_range test_re
test_struct test_array test_binascii test_base64 test_hashlib test_csv test_textwrap test_difflib test_pprint test_reprlib
test_copy test_marshal test_dis test_tokenize test_unparse test_format test_fstring test_codecs test_collections
test_contextlib test_dataclasses test_enum test_abc test_inspect test_traceback test_keyword test_numeric_tower
test_decorators test_augassign test_bool test_builtin test_code_module test_colorsys test_configparser test_contains
test_copyreg test_cprofile test_dictcomps test_dictviews test_ordered_dict test_defaultdict test_genericalias test_getopt
test_userstring test_graphlib test_glob test_html test_htmlparser test_ipaddress test_iter test_keywordonlyarg
test_linecache test_locale test_minidom test_mmap test_named_expressions test_pep646_syntax test_mailbox
test_positional_only_arg test_print test_property test_raise test_setcomps test_shlex test_string_literals
test_stringprep test_sqlite3 test_sundry test_super test_strtod test_tabnanny test_thread test_timeit test_tomllib
test_time test_typechecks test_unary test_unpack test_unpack_ex test_utf8source test_uuid test_urlparse test_xml_etree
test_zlib test_bufio test_fnmatch test_gettext test_int_literal test_longexp test_mimetypes test_pickletools
test_plistlib test_quopri test_richcmp test_rlcompleter test_secrets test_unicodedata""".split()

_memo_lock = threading.Lock()
_memo = {}


def memo(key, fn):
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
        return {"units": [], "modules": {}}
    with open(p) as f:
        return json.load(f)


def slug(key):
    m = re.match(r"(\S+) \(([\w.]+)\)", key)
    name = ".".join(m.group(2).split(".")[2:]) if m else key
    return re.sub(r"[^A-Za-z0-9_]+", "-", name).strip("-")[:60] or "case"


def list_tests(suite):
    m = manifest(suite)
    out = ["src/%s.c" % u for u in m["units"]]
    for mod, info in sorted(m["modules"].items()):
        out += ["case/%s/%04d_%s" % (mod, idx, sl) for idx, sl in info["cases"]]
    return out


# ------------------------------------------------------------------ the reference tree

WRAPPER = """#!/bin/bash
# records compile commands (cwd and arguments, 0x1f separated), then runs gcc
for a in "$@"; do if [ "$a" = "-c" ]; then printf '%s\\x1f' "$PWD" >> "$CCLOG"; printf '%s\\x1f' "$@" >> "$CCLOG"; printf '\\n' >> "$CCLOG"; break; fi; done
exec gcc "$@"
"""


def tree():
    """(tree dir, {unit: (source path, [flags])}) of the gcc-built reference tree."""
    src = P.fetch_git("cpython", URL, REF, COMMIT)
    base = os.path.join(C.cache_root(), "cpython-tree", C.sha(COMMIT, "v1"))
    t = os.path.join(base, "t")
    with C.locked(base):
        if not os.path.exists(os.path.join(base, "built")):
            wrap, log = os.path.join(base, "ccwrap.sh"), os.path.join(base, "cc.log")
            rebuilt = os.path.exists(os.path.join(t, "python")) and os.path.exists(log)  # an earlier attempt got this far
            if not rebuilt:
                shutil.rmtree(t, ignore_errors=True)
                os.makedirs(base, exist_ok=True)
                shutil.copytree(src, t, ignore=shutil.ignore_patterns(".git", "Doc"), symlinks=True)
            with open(wrap, "w") as f:
                f.write(WRAPPER)
            os.chmod(wrap, 0o755)
            env = dict(os.environ, CCLOG=log)
            if not rebuilt:
                sh(["./configure", "CC=" + wrap, "CFLAGS=-O0"], t, env)
                if os.path.exists(log):
                    os.remove(log)
                sh(["make", "-j2"], t, env)
            objs = sh(["make", "-s", "--eval=.PHONY: pobjs\npobjs: ; @echo $(LINK_PYTHON_OBJS)", "pobjs"], t, env).split()
            units = extract_units(log, t, objs + ["Programs/python.o"])
            with open(os.path.join(base, "units.json"), "w") as f:
                json.dump(units, f)
            for name, (srcp, flags) in list(units.items()):  # assembler sources are always built by gcc
                if srcp.endswith(".S"):
                    extra = os.path.join(base, "extra")
                    os.makedirs(extra, exist_ok=True)
                    sh(["gcc", "-c", "-O0", "-o", os.path.join(extra, name.replace("/", "_") + ".o")] + flags + [srcp], t, os.environ)
                    del units[name]
            with open(os.path.join(base, "units.json"), "w") as f:
                json.dump(units, f)
            open(os.path.join(base, "built"), "w").close()
    with open(os.path.join(base, "units.json")) as f:
        return t, json.load(f)


def extra_objects():
    base = os.path.join(C.cache_root(), "cpython-tree", C.sha(COMMIT, "v1"), "extra")
    return sorted(os.path.join(base, f) for f in os.listdir(base)) if os.path.isdir(base) else []


def sh(cmd, cwd, env):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("cpython tree: %s failed:\n%s" % (" ".join(cmd[:3]), (r.stdout + r.stderr)[-2000:]))
    return r.stdout


def extract_units(log, t, objs):
    """Last recorded compile command for each wanted object -> {unit: [source relative to t, flags]}."""
    rec = {}
    for line in open(log, "rb").read().split(b"\n"):
        if not line:
            continue
        parts = line.decode().split("\x1f")[:-1]
        cwd, args = parts[0], parts[1:]
        if "-o" not in args:
            continue
        out = os.path.relpath(os.path.normpath(os.path.join(cwd, args[args.index("-o") + 1])), t)
        if out in objs:
            rec[out] = (cwd, args)
    units = {}
    for o in objs:
        cwd, args = rec[o]
        srcs = [a for a in args if a.endswith((".c", ".S"))]
        flags = []
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-o",):
                i += 2
                continue
            if a.startswith("-I") and len(a) > 2 and not os.path.isabs(a[2:]):
                flags.append("-I" + os.path.normpath(os.path.join(cwd, a[2:])))
            elif a.startswith(("-D", "-U", "-std=")) or a in ("-pthread",):
                flags.append(a)
            elif a == "-include" or a == "-isystem":
                flags += [a, args[i + 1]]
                i += 1
            i += 1
        units[o[:-2]] = [os.path.relpath(os.path.normpath(os.path.join(cwd, srcs[0])), t), flags]
    return units


def project(R, suite, cc_argv):
    t, units = tree()
    paths = {u: os.path.join(t, s) for u, (s, f) in units.items()}
    uflags = {u: f for u, (s, f) in units.items()}
    proj = P.Project(R, suite, cc_argv, "cpython", paths, [], LDFLAGS, uflags)
    proj.extra_objs = extra_objects()
    return proj, t, sorted(units)


# ------------------------------------------------------------------ running

def pyhome(t, exe):
    """A directory laid out like the build tree around a hard link to exe, so that python finds Lib and the
    gcc-built extension modules."""
    d = os.path.join(C.cache_root(), "cpython-run", C.file_sha(exe))
    with C.locked(d):
        if not os.path.exists(os.path.join(d, "python")):
            os.makedirs(os.path.join(d, "Modules"), exist_ok=True)
            open(os.path.join(d, "Modules", "Setup.local"), "w").close()
            shutil.copy(os.path.join(t, "pybuilddir.txt"), d)
            for n in ("Lib", "build"):
                if not os.path.lexists(os.path.join(d, n)):
                    os.symlink(os.path.join(t, n), os.path.join(d, n))
            try:
                os.link(exe, os.path.join(d, "python.tmp"))
            except OSError:
                shutil.copy(exe, os.path.join(d, "python.tmp"))
            os.replace(os.path.join(d, "python.tmp"), os.path.join(d, "python"))
    return os.path.join(d, "python")


def parse(text):
    keys, status = [], {}
    for m in LINE.finditer(text):
        k = "%s (%s)" % (m.group(1), m.group(2))
        n = 1
        while k in status:
            n += 1
            k = "%s (%s)#%d" % (m.group(1), m.group(2), n)
        keys.append(k)
        st = m.group(3)
        status[k] = "skipped" if st.startswith("skipped") else st
    return keys, status


def run_module(R, t, exe, module, work):
    py = pyhome(t, exe)
    d = P.tmpdir(work, "run-" + module)
    env = dict(os.environ, PYTHONHASHSEED="0", LC_ALL="C.UTF-8", HOME=d)
    env.pop("PYTHONPATH", None)
    rc, out, err, to = R.run_cmd([py, "-m", "test", "-v", module], RUN_TIMEOUT, cwd=d, env=env)
    text = R.text(out, 8000000)
    keys, status = parse(text)
    return {"rc": rc, "timeout": to, "keys": keys, "status": status, "tail": (text[-300:] + R.text(err)[-300:])}


def cached_run(R, t, exe, module, work, who):
    key = C.sha(who, C.file_sha(exe), module, C.file_sha(__file__))
    p = os.path.join(C.cache_root(), "val", "cpython-run", key[:2], key + ".json")
    with C.locked(p):
        v = C.read_json(p)
        if v is not None:
            return v["v"]
        v = run_module(R, t, exe, module, work)
        if not v["timeout"]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            C.atomic_write(p, json.dumps({"v": v}))
        return v


def reference(R, proj, t, units, module, work):
    ref = proj.ref("python", units, LDFLAGS)
    return cached_run(R, t, ref, module, work, "ref")


def compare_case(R, got, want, idx):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "module run >%ds" % RUN_TIMEOUT)
    key = want["keys"][idx]
    st = got["status"].get(key)
    if st is not None:
        return R.Result("PASS") if st == want["status"][key] else R.Result("FAIL", "compare", "%s, expected %s" % (st, want["status"][key]))
    if got["rc"] < 0:
        return R.Result("FAIL", "run", R.short("python killed by %s after %d of %d cases %s" % (R.signame(got["rc"]), len(got["keys"]), len(want["keys"]), got["tail"][-100:])))
    return R.Result("FAIL", "run", R.short("no result line for the case (exit %d, %d of %d cases) %s" % (got["rc"], len(got["keys"]), len(want["keys"]), got["tail"][-120:])))


def compare_module(R, got, want, module):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "module run >%ds" % RUN_TIMEOUT)
    for k in want["keys"]:
        if got["status"].get(k) != want["status"][k]:
            if got["rc"] < 0 and k not in got["status"]:
                return R.Result("FAIL", "run", R.short("%s: python killed by %s %s" % (module, R.signame(got["rc"]), got["tail"][-100:])))
            return R.Result("FAIL", "compare", R.short("%s: %s is %s, expected %s" % (module, k, got["status"].get(k, "missing"), want["status"][k])))
    return R.Result("PASS")


def run_test(suite, rel, cc_argv, work, R):
    cc = tuple(cc_argv)
    proj, t, units = memo(("proj", cc), lambda: project(R, suite, cc_argv))
    kind, _, rest = rel.partition("/")
    if kind == "case":
        module, _, cs = rest.partition("/")
        idx = int(cs.split("_", 1)[0])
        exe, fail = memo(("prog", cc), lambda: proj.full("python", units, LDFLAGS, degrade=True))
        if fail:
            return fail.result(R)
        want = memo(("ref", module), lambda: reference(R, proj, t, units, module, work))
        got = memo(("run", cc, module), lambda: cached_run(R, t, exe, module, work, "hcc"))
        return compare_case(R, got, want, idx)
    unit = rest[:-2]
    out = os.path.join(work, "python-mixed")
    f = proj.mixed(out, units, [unit], LDFLAGS)
    if f:
        return f.result(R)
    for module in SMOKE:
        want = memo(("ref", module), lambda: reference(R, proj, t, units, module, work))
        res = compare_module(R, run_module(R, t, out, module, P.tmpdir(work, "got")), want, module)
        if res.status != "PASS":
            return res
    return R.Result("PASS")


# ------------------------------------------------------- manifest generation

def build_manifest(R, suite):
    """Used by tests/scripts/gen_cpython_manifest.py."""
    import tempfile
    work = tempfile.mkdtemp(prefix="cpy-manifest-")
    proj, t, units = project(R, suite, ["gcc"])
    ref = proj.ref("python", units, LDFLAGS)
    out = {"units": units, "modules": {}}
    for module in CANDIDATES:
        a = run_module(R, t, ref, module, P.tmpdir(work, "a"))
        b = run_module(R, t, ref, module, P.tmpdir(work, "b"))
        if a["timeout"] or a["rc"] != 0 or b["rc"] != 0:
            print("skipping %s: rc=%s/%s timeout=%s" % (module, a["rc"], b["rc"], a["timeout"]))
            continue
        live = [(i, slug(k)) for i, k in enumerate(a["keys"]) if a["status"][k] == "ok" and b["status"].get(k) == "ok"]
        if not live:
            continue
        stride = -(-len(live) // MAX_LIVE)
        out["modules"][module] = {"total": len(a["keys"]), "live": len(live), "stride": stride, "cases": live[::stride]}
    shutil.rmtree(work, ignore_errors=True)
    return out
