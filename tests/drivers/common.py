"""Shared helpers for suite drivers (tests/drivers/*.py).

A driver builds a real program (Lua, SQLite, QuickJS) with the compiler under
test and compares its behaviour with a gcc-built copy of the same program.  The
expensive parts are cached on disk so that "one test per script/file" stays
cheap:

* object files compiled by the compiler under test are cached per
  (compiler fingerprint, source bytes, flags);
* linked programs are cached per (compiler fingerprint, program recipe);
* the gcc-built reference program and its per-test outputs are cached per
  (gcc version, source bytes).

The compiler fingerprint is a hash of every hcc/**/*.py file, so any change to
hcc invalidates the hcc-built artefacts (and only those).  Concurrent test
threads and processes serialise on per-key flock()s, so each artefact is built
once per run.

Cache root: $HCC_TEST_CACHE, else <main checkout>/tests/cache.  Downloaded
third-party sources: $HCC_TEST_TOOLS, else <main checkout>/tests/tools.  The main
checkout is found through `git rev-parse --git-common-dir`, so swarm worktrees
share one cache instead of rebuilding in each worktree.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TESTS_DIR)


def main_checkout():
    try:
        out = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--git-common-dir"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            common = os.path.abspath(os.path.join(REPO_ROOT, out))
            top = os.path.dirname(common)
            if os.path.isdir(os.path.join(top, "tests")):
                return top
    except Exception:
        pass
    return REPO_ROOT


def cache_root():
    d = os.environ.get("HCC_TEST_CACHE") or os.path.join(main_checkout(), "tests", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def tools_root():
    d = os.environ.get("HCC_TEST_TOOLS") or os.path.join(main_checkout(), "tests", "tools")
    os.makedirs(d, exist_ok=True)
    return d


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            p = p.encode()
        h.update(str(len(p)).encode() + b":")
        h.update(p)
    return h.hexdigest()[:20]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


_fp_cache = {}


def compiler_fingerprint(R, cc_argv):
    """Identify the compiler under test: gcc version for a real cc, else a hash
    of all hcc python sources plus the command line."""
    key = tuple(cc_argv)
    if key in _fp_cache:
        return _fp_cache[key]
    if R.is_real_cc(cc_argv):
        ver = subprocess.run(cc_argv + ["--version"], capture_output=True, text=True).stdout
        fp = "real-" + sha(ver.splitlines()[0] if ver else "?", " ".join(cc_argv))
    else:
        h = hashlib.sha256(" ".join(cc_argv).encode())
        root = os.path.join(REPO_ROOT, "hcc")
        for dp, dn, fn in os.walk(root):
            dn.sort()
            for f in sorted(fn):
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    h.update(os.path.relpath(p, root).encode())
                    with open(p, "rb") as fh:
                        h.update(fh.read())
        fp = "hcc-" + h.hexdigest()[:20]
    _fp_cache[key] = fp
    return fp


def gcc_fingerprint():
    ver = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout
    return "gcc-" + sha(ver.splitlines()[0] if ver else "?")


@contextlib.contextmanager
def locked(path):
    """Exclusive inter-thread / inter-process lock on <path>.lock."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write(path, data):
    mode = "wb" if isinstance(data, bytes) else "w"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    with os.fdopen(fd, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------ compile steps

class Failure:
    """A failed build step.  Maps onto R.Result for the runner."""

    def __init__(self, status, stage, reason, log=""):
        self.status, self.stage, self.reason, self.log = status, stage, reason, log

    def to_json(self):
        return {"status": self.status, "stage": self.stage, "reason": self.reason, "log": self.log[-6000:]}

    @classmethod
    def from_json(cls, d):
        return cls(d["status"], d["stage"], d["reason"], d.get("log", ""))

    def result(self, R, prefix=""):
        return R.Result(self.status, self.stage, R.short(prefix + self.reason), self.log)


def compile_c(R, cc_argv, cflags, src, obj, cwd, timeout=900):
    """src -> obj with the given compiler (cc -S, then as).  Returns None or Failure."""
    asm = obj[:-2] + ".s" if obj.endswith(".o") else obj + ".s"
    rc, _, err, to = R.run_cmd(cc_argv + cflags + ["-S", "-o", asm, src], timeout,
                               cwd=cwd, env=R.cc_env())
    if to:
        return Failure("TIMEOUT", "timeout", "compile >%ds" % timeout)
    if rc != 0 or not os.path.exists(asm):
        if rc == 0:
            err += b"\n(no assembly output produced)"
            rc = 1
        res = R.compile_failure(cc_argv, cflags, src, cwd, rc, err)
        return Failure(res.status, res.stage, res.reason, res.log)
    rc, _, err, to = R.run_cmd(["as", "--64", "-o", obj, asm], 300, cwd=cwd)
    with contextlib.suppress(OSError):
        os.remove(asm)
    if to or rc != 0:
        e = R.text(err)
        lines = [l for l in e.splitlines() if "Error" in l] or e.splitlines() or ["as failed"]
        return Failure("FAIL", "assemble", R.short(lines[0].split(": ", 1)[-1]), e)
    return None


def cached_object(R, cc_argv, cflags, src, tag, timeout=900, extra_key=""):
    """Compile `src` with the compiler under test, cached.  Returns (obj_path, None)
    or (None, Failure).  `tag` is a short human label used in the cache dir name."""
    key = sha(compiler_fingerprint(R, cc_argv), file_sha(src), " ".join(cflags), extra_key)
    d = os.path.join(cache_root(), "obj", key[:2], key)
    obj = os.path.join(d, "out.o")
    failp = os.path.join(d, "fail.json")
    with locked(d):
        if os.path.exists(obj):
            return obj, None
        f = read_json(failp)
        if f is not None:
            return None, Failure.from_json(f)
        os.makedirs(d, exist_ok=True)
        work = tempfile.mkdtemp(prefix="cc-", dir=d)
        try:
            tmp_obj = os.path.join(work, "out.o")
            fail = compile_c(R, cc_argv, cflags, src, tmp_obj, work, timeout)
            if fail is None:
                os.replace(tmp_obj, obj)
                atomic_write(os.path.join(d, "label"), tag + "\n")
                return obj, None
            if fail.status == "TIMEOUT" or "runner error" in fail.reason:
                return None, fail  # do not cache timeouts: load dependent
            atomic_write(failp, json.dumps(fail.to_json()))
            atomic_write(os.path.join(d, "label"), tag + "\n")
            return None, fail
        finally:
            shutil.rmtree(work, ignore_errors=True)


def gcc_object(cflags, src, tag):
    """Reference object built by host gcc -O0, cached."""
    key = sha(gcc_fingerprint(), file_sha(src), " ".join(cflags))
    d = os.path.join(cache_root(), "gccobj", key[:2], key)
    obj = os.path.join(d, "out.o")
    with locked(d):
        if os.path.exists(obj):
            return obj
        os.makedirs(d, exist_ok=True)
        tmp = obj + ".tmp"
        r = subprocess.run(["gcc", "-O0", "-w", "-x", "c"] + cflags + ["-c", "-o", tmp, src],
                           capture_output=True, text=True, cwd=d)
        if r.returncode != 0:
            raise RuntimeError("reference gcc build of %s failed:\n%s" % (src, r.stderr[-2000:]))
        os.replace(tmp, obj)
        return obj


def link(R, out, objs, ldflags, cwd, timeout=300):
    """gcc -no-pie link.  Returns None or Failure(stage=link)."""
    rc, _, err, to = R.run_cmd(["gcc", "-no-pie", "-o", out] + list(objs) + list(ldflags),
                               timeout, cwd=cwd)
    if to or rc != 0:
        e = R.text(err)
        lines = [l for l in e.splitlines() if "undefined reference" in l or "error" in l.lower()]
        reason = lines[0] if lines else (e.strip().splitlines() or ["link failed"])[0]
        reason = re.sub(r"^.*?: ", "", reason) if "undefined reference" in reason else reason
        return Failure("FAIL", "link", R.short(reason), e)
    return None


def cached_program(R, name, key_parts, build):
    """Cache a linked program (or any single output file).  build(out_path) must
    return None on success or a Failure.  Returns (path, None) or (None, Failure)."""
    key = sha(*key_parts)
    d = os.path.join(cache_root(), "prog", name, key)
    exe = os.path.join(d, "out")
    failp = os.path.join(d, "fail.json")
    with locked(d):
        if os.path.exists(exe):
            return exe, None
        f = read_json(failp)
        if f is not None:
            return None, Failure.from_json(f)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "out.tmp")
        fail = build(tmp)
        if fail is None:
            os.replace(tmp, exe)
            return exe, None
        if fail.status != "TIMEOUT":
            atomic_write(failp, json.dumps(fail.to_json()))
        return None, fail


def cached_value(name, key_parts, compute):
    """Cache a JSON-serialisable value (reference outputs)."""
    key = sha(*key_parts)
    p = os.path.join(cache_root(), "val", name, key[:2], key + ".json")
    with locked(p):
        v = read_json(p)
        if v is not None:
            return v["v"]
        v = compute()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        atomic_write(p, json.dumps({"v": v}))
        return v


# ----------------------------------------------------------------- fetching

def fetch(url, dest, sha256_prefix=None):
    """Download url to dest (atomic).  Used for large third-party sources that are
    not committed (they live in tests/tools, which is gitignored)."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with locked(dest):
        if os.path.exists(dest):
            return dest
        tmp = dest + ".part"
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if sha256_prefix:
            got = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
            if not got.startswith(sha256_prefix):
                os.remove(tmp)
                raise RuntimeError("checksum mismatch for %s: %s" % (url, got))
        os.replace(tmp, dest)
    return dest


# ------------------------------------------------------------------ running

def run_and_capture(R, argv, timeout, cwd, env=None, stdin_data=None):
    rc, out, err, to = R.run_cmd(argv, timeout, cwd=cwd, env=env, stdin_data=stdin_data)
    return {"rc": rc, "out": R.text(out), "err": R.text(err), "timeout": to}


def normalise(s, patterns):
    for pat, rep in patterns:
        s = re.sub(pat, rep, s)
    return s


def compare_runs(R, got, want, what="stdout"):
    """Compare run dicts {rc,out,timeout}.  Returns a Result."""
    last = (got["err"].strip().splitlines() or got["out"].strip().splitlines() or [""])[-1]
    log = "exit %s (want %s)\n--- stdout ---\n%s\n--- stderr ---\n%s\n--- expected stdout ---\n%s" % (
        got["rc"], want["rc"], got["out"][-6000:], got["err"][-3000:], want["out"][-6000:])
    if got["timeout"]:
        return R.Result("TIMEOUT", "timeout", "run timeout", log)
    if got["rc"] < 0:
        what_ = "abort()" if -got["rc"] == 6 else R.signame(got["rc"])
        return R.Result("FAIL", "run", R.short("killed by %s %s" % (what_, last)), log)
    if got["rc"] != want["rc"]:
        return R.Result("FAIL", "compare", R.short("exit %d, expected %d %s" % (got["rc"], want["rc"], last)), log)
    if got["out"] != want["out"]:
        return R.Result("FAIL", "compare",
                        R.short(R.describe_diff(want["out"].encode(), got["out"].encode())), log)
    return R.Result("PASS", None, "", log)
