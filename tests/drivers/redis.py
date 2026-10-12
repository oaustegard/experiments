"""Driver for the `redis` suite: Redis 7.2.5 (git tag, commit-pinned) built by hcc with MALLOC=libc.

The tree is built once by gcc (-O0, `make MALLOC=libc`) in a private copy under tests/cache/redis-tree, with a CC
wrapper recording every compile command: that gives the reference redis-server / redis-cli / redis-benchmark and,
for each object of redis-server (the 100 files of src/ plus the hiredis, Lua 5.1, hdr_histogram and fpconv objects of
deps/), the exact flags.  hcc compiles those objects and gcc links redis-server; redis-cli, redis-benchmark and
the check tools of the run directory are always the gcc-built ones.  Tests are files of Redis' own Tcl suite run
by `./runtest --single <file> --clients 1` in a private copy of tests/ (short paths: unix sockets); each
`[ok]: <name>` line is one test case, which passes when the gcc-built server also passes it.

Test ids (tests/suites/redis/manifest.json; regenerate with tests/scripts/gen_redis_manifest.py):
  redis/case/<file>/<nnnn>_<slug>   one test case of a Tcl test file (file path with / replaced by .), run against
                       a redis-server whose objects are all built by hcc, except that an object hcc cannot compile
                       yet is replaced by its gcc object.  Cases that pass with the gcc build in two runs only.
  redis/src/<dir>/<file>.c   hcc compiles that one translation unit of redis-server; the rest is gcc; the smoke
                       files unit.type.incr and unit.type.string must give the gcc build's status for every case.
"""
import json
import os
import random
import re
import shutil
import subprocess
import threading

import common as C
import projlib as P

URL = "https://github.com/redis/redis.git"
REF = "7.2.5"
COMMIT = "f60370ce28b946c1146dcea77c9c399d39601aaa"
SMOKE = ["unit.type.incr", "unit.type.string"]
RUN_TIMEOUT = 900
RUN_VERSION = "1"  # bump when run_file / parse change: cached reference runs are keyed on it
MAX_LIVE = 400
LDFLAGS = ["-lm", "-ldl", "-pthread"]
LINE = re.compile(r"^\[(ok|err|skip|ignore|exception)\]:\s*(.*?)(?: \(\d+ ms\))?\s*$", re.M)
CANDIDATES = """unit/acl unit/acl-v2 unit/auth unit/bitfield unit/bitops unit/client-eviction unit/dump unit/expire
unit/functions unit/info unit/info-command unit/introspection unit/introspection-2 unit/keyspace unit/latency-monitor
unit/lazyfree unit/limits unit/memefficiency unit/multi unit/networking unit/obuf-limits unit/other unit/pause
unit/protocol unit/pubsub unit/pubsubshard unit/querybuf unit/replybufsize unit/scan unit/scripting unit/slowlog
unit/sort unit/tracking unit/type/hash unit/type/incr unit/type/list unit/type/list-2 unit/type/set unit/type/stream
unit/type/stream-cgroups unit/type/string unit/type/zset""".split()

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
        return {"units": [], "files": {}}
    with open(p) as f:
        return json.load(f)


def slug(name):
    return re.sub(r"[^A-Za-z0-9_]+", "-", name).strip("-")[:48] or "case"


def list_tests(suite):
    m = manifest(suite)
    out = ["src/%s.c" % u for u in m["units"]]
    for f, info in sorted(m["files"].items()):
        out += ["case/%s/%04d_%s" % (f, idx, sl) for idx, sl in info["cases"]]
    return out


# ------------------------------------------------------------------ the reference tree

WRAPPER = """#!/bin/bash
# records compile commands (cwd and arguments, 0x1f separated), then runs gcc
for a in "$@"; do if [ "$a" = "-c" ]; then printf '%s\\x1f' "$PWD" >> "$CCLOG"; printf '%s\\x1f' "$@" >> "$CCLOG"; printf '\\n' >> "$CCLOG"; break; fi; done
exec gcc "$@"
"""
DEP_ARCHIVES = [("deps/hiredis", "libhiredis.a"), ("deps/lua/src", "liblua.a"), ("deps/hdr_histogram", "libhdrhistogram.a"), ("deps/fpconv", "libfpconv.a")]


def tree():
    """(tree dir, {unit: [source relative to tree, flags]}) of the gcc-built reference tree."""
    src = P.fetch_git("redis", URL, REF, COMMIT)
    base = os.path.join(C.cache_root(), "redis-tree", C.sha(COMMIT, "v1"))
    t = os.path.join(base, "t")
    with C.locked(base):
        if not os.path.exists(os.path.join(base, "built")):
            wrap, log = os.path.join(base, "ccwrap.sh"), os.path.join(base, "cc.log")
            done = os.path.exists(os.path.join(t, "src", "redis-server")) and os.path.exists(log)  # an earlier attempt built it
            if not done:
                shutil.rmtree(t, ignore_errors=True)
                os.makedirs(base, exist_ok=True)
                shutil.copytree(src, t, ignore=shutil.ignore_patterns(".git"), symlinks=True)
            with open(wrap, "w") as f:
                f.write(WRAPPER)
            os.chmod(wrap, 0o755)
            env = dict(os.environ, CCLOG=log)
            if not done:
                if os.path.exists(log):
                    os.remove(log)
                r = subprocess.run(["make", "-j2", "MALLOC=libc", "OPTIMIZATION=-O0", "CC=" + wrap], cwd=t, env=env, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError("redis tree: make failed:\n%s" % (r.stdout + r.stderr)[-2000:])
            wanted = []
            mk = open(os.path.join(t, "src", "Makefile")).read()
            for o in re.search(r"^REDIS_SERVER_OBJ=(.*)$", mk, re.M).group(1).split():
                wanted.append("src/" + o)
            for d, a in DEP_ARCHIVES:
                for m in subprocess.run(["ar", "t", os.path.join(t, d, a)], capture_output=True, text=True).stdout.split():
                    wanted.append(d + "/" + m)
            units = extract_units(log, t, wanted)
            with open(os.path.join(base, "units.json"), "w") as f:
                json.dump(units, f)
            open(os.path.join(base, "built"), "w").close()
    with open(os.path.join(base, "units.json")) as f:
        return t, json.load(f)


def extract_units(log, t, wanted):
    rec = {}
    for line in open(log, "rb").read().split(b"\n"):
        if not line:
            continue
        parts = line.decode().split("\x1f")[:-1]
        cwd, args = parts[0], parts[1:]
        srcs = [a for a in args if a.endswith(".c")]
        if "-o" in args:
            outarg = args[args.index("-o") + 1]
        elif srcs:
            outarg = os.path.basename(srcs[0])[:-2] + ".o"  # gcc -c without -o
        else:
            continue
        out = os.path.relpath(os.path.normpath(os.path.join(cwd, outarg)), t)
        if out in wanted:
            rec[out] = (cwd, args)
    units = {}
    for o in wanted:
        cwd, args = rec[o]
        srcs = [a for a in args if a.endswith(".c")]
        flags, i = [], 0
        while i < len(args):
            a = args[i]
            if a == "-o":
                i += 2
                continue
            if a.startswith("-I") and len(a) > 2 and not os.path.isabs(a[2:]):
                flags.append("-I" + os.path.normpath(os.path.join(cwd, a[2:])))
            elif a.startswith(("-D", "-U", "-std=")) or a == "-pthread":
                flags.append(a)
            elif a in ("-include", "-isystem"):
                flags += [a, args[i + 1]]
                i += 1
            i += 1
        units[o[:-2]] = [os.path.relpath(os.path.normpath(os.path.join(cwd, srcs[0])), t), flags]
    return units


def project(R, suite, cc_argv):
    t, units = tree()
    paths = {u: os.path.join(t, s) for u, (s, f) in units.items()}
    uflags = {u: f for u, (s, f) in units.items()}
    return P.Project(R, suite, cc_argv, "redis", paths, [], LDFLAGS, uflags), t, sorted(units)


# ------------------------------------------------------------------ running

def parse(text):
    names, status = [], {}
    for m in LINE.finditer(text):
        k = m.group(2)
        n = 1
        base = k
        while k in status:
            n += 1
            k = "%s#%d" % (base, n)
        names.append(k)
        status[k] = m.group(1)
    return names, status


def run_file(R, t, exe, tfile, work):
    """tfile like 'unit.type.incr'.  Short paths: the unix socket of a test server lives under the run directory."""
    d = P.tmpdir(work, "r")
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    for b in ("redis-cli", "redis-benchmark"):
        link = os.path.join(d, "src", b)
        if not os.path.lexists(link):
            os.symlink(os.path.join(t, "src", b), link)
    for b in ("redis-server", "redis-check-aof", "redis-check-rdb", "redis-sentinel"):
        link = os.path.join(d, "src", b)
        if not os.path.lexists(link):
            os.symlink(exe, link)
    if not os.path.exists(os.path.join(d, "tests")):
        shutil.copytree(os.path.join(t, "tests"), os.path.join(d, "tests"), ignore=shutil.ignore_patterns("tmp"))
        os.makedirs(os.path.join(d, "tests", "tmp"))
        os.symlink(os.path.join(t, "utils"), os.path.join(d, "utils"))
        for f in ("runtest", "redis.conf", "sentinel.conf"):
            shutil.copy(os.path.join(t, f), d)
    port = random.randrange(20000, 52000, 100)
    rc, out, err, to = P.run_cmd(R, ["sh", "./runtest", "--single", tfile.replace(".", "/"), "--clients", "1", "--baseport", str(port),
                                   "--portcount", "50", "--timeout", "800"], RUN_TIMEOUT, cwd=d)
    text = R.text(out, 8000000)
    names, status = parse(text)
    return {"rc": rc, "timeout": to, "names": names, "status": status, "tail": text[-300:] + R.text(err)[-200:]}


def cached_run(R, t, exe, tfile, work, who):
    key = C.sha(who, C.file_sha(exe), tfile, RUN_VERSION)
    p = os.path.join(C.cache_root(), "val", "redis-run", key[:2], key + ".json")
    with C.locked(p):
        v = C.read_json(p)
        if v is not None:
            return v["v"]
        v = run_file(R, t, exe, tfile, work)
        if not v["timeout"]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            C.atomic_write(p, json.dumps({"v": v}))
        return v


def reference(R, proj, t, tfile, work):
    return cached_run(R, t, os.path.join(t, "src", "redis-server"), tfile, work, "ref")


def compare_case(R, got, want, idx):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "test file run >%ds" % RUN_TIMEOUT)
    name = want["names"][idx]
    st = got["status"].get(name)
    if st is not None:
        return R.Result("PASS") if st == want["status"][name] else R.Result("FAIL", "compare", "[%s], expected [%s]" % (st, want["status"][name]))
    if got["rc"] < 0:
        return R.Result("FAIL", "run", R.short("test run killed by %s after %d of %d cases" % (R.signame(got["rc"]), len(got["names"]), len(want["names"]))))
    return R.Result("FAIL", "run", R.short("case did not run (%d of %d cases ran, exit %d) %s" % (len(got["names"]), len(want["names"]), got["rc"], got["tail"][-150:])))


def compare_file(R, got, want, what):
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "test file run >%ds" % RUN_TIMEOUT)
    for k in want["names"]:
        if got["status"].get(k) != want["status"][k]:
            return R.Result("FAIL", "compare", R.short("%s: %s is %s, expected %s %s" % (what, k, got["status"].get(k, "missing"), want["status"][k], got["tail"][-100:])))
    return R.Result("PASS")


def run_test(suite, rel, cc_argv, work, R):
    cc = tuple(cc_argv)
    proj, t, units = memo(("proj", cc), lambda: project(R, suite, cc_argv))
    kind, _, rest = rel.partition("/")
    if kind == "case":
        tfile, _, cs = rest.partition("/")
        idx = int(cs.split("_", 1)[0])
        exe, fail = memo(("prog", cc), lambda: proj.full("redis-server", units, LDFLAGS, degrade=True))
        if fail:
            return fail.result(R)
        want = memo(("ref", tfile), lambda: reference(R, proj, t, tfile, work))
        got = memo(("run", cc, tfile), lambda: cached_run(R, t, exe, tfile, work, "hcc"))
        return compare_case(R, got, want, idx)
    unit = rest[:-2]
    out = os.path.join(work, "redis-server")
    f = proj.mixed(out, units, [unit], LDFLAGS)
    if f:
        return f.result(R)
    for tfile in SMOKE:
        want = memo(("ref", tfile), lambda: reference(R, proj, t, tfile, work))
        res = compare_file(R, run_file(R, t, out, tfile, P.tmpdir(work, "got-" + tfile)), want, tfile)
        if res.status != "PASS":
            return res
    return R.Result("PASS")


# ------------------------------------------------------- manifest generation

def build_manifest(R, suite):
    """Used by tests/scripts/gen_redis_manifest.py."""
    import tempfile
    work = tempfile.mkdtemp(prefix="redis-manifest-")
    proj, t, units = project(R, suite, ["gcc"])
    out = {"units": units, "files": {}}
    for f in CANDIDATES:
        tf = f.replace("/", ".")
        a = run_file(R, t, os.path.join(t, "src", "redis-server"), tf, P.tmpdir(work, "a"))
        b = run_file(R, t, os.path.join(t, "src", "redis-server"), tf, P.tmpdir(work, "b"))
        if a["timeout"] or a["rc"] != 0 or b["rc"] != 0:
            print("skipping %s: rc=%s/%s timeout=%s" % (f, a["rc"], b["rc"], a["timeout"]))
            continue
        live = [(i, slug(k)) for i, k in enumerate(a["names"]) if a["status"][k] == "ok" and b["status"].get(k) == "ok"]
        if not live:
            continue
        stride = -(-len(live) // MAX_LIVE)
        out["files"][tf] = {"total": len(a["names"]), "live": len(live), "stride": stride, "cases": live[::stride]}
    shutil.rmtree(work, ignore_errors=True)
    return out
