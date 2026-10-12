#!/usr/bin/env python3
"""hcc test runner.

Usage:
    python3 tests/run.py --suite smoke [--jobs 4] [--filter GLOB] [--json OUT]
                         [--cc 'python3 -m hcc'] [--verbose] [--keep DIR]
                         [--baseline OLD.json] [--list] [--include-skipped]

Every test is a single C file. Pipeline per test (each step is a "stage"):
    1. compile   <cc> <suite cflags> -S -o t.s test.c   (stages preprocess/parse/sema/codegen)
    2. assemble  as --64 -o t.o t.s                    (stage assemble)
    3. link      gcc -no-pie -o t t.o <support objs> <ldflags>   (stage link)
    4. run       ./t   with stdin=/dev/null, 10s timeout (stage run: signal / crash)
    5. compare   exit code and stdout vs expectations  (stage compare)

Output: one line per test "PASS|FAIL|TIMEOUT|CRASH <test-id> <short reason>",
then a SUMMARY line.  --json writes {test_id: {status, reason, stage}}.
CRASH means the *compiler* crashed (Python traceback or signal).
Suite layout and suite.json keys are documented in tests/README.md.
"""
import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SUITES_DIR = os.path.join(TESTS_DIR, "suites")

COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 10
TOOL_TIMEOUT = 60  # as / ld

STAGES = ("preprocess", "parse", "sema", "codegen", "assemble", "link",
          "run", "compare", "timeout")

# Suite config defaults; see tests/README.md.
SUITE_DEFAULTS = {
    "description": "",
    "glob": ["*.c"],          # test files, relative to the suite dir
    "cflags": [],             # passed to --cc for every test
    "gcc_cflags": [],         # extra flags only when --cc is a real gcc/clang
    "support": [],            # extra C files compiled by host gcc and linked in
    "ldflags": ["-lm"],
    "skip_file": "skip.txt",  # lines: "<relative path>  # reason"
    "expect_exit": 0,         # default expected exit code
}


# ---------------------------------------------------------------- helpers

class Result:
    """Outcome of one test.  status in PASS/FAIL/TIMEOUT/CRASH."""

    def __init__(self, status, stage=None, reason="", log=""):
        self.status = status
        self.stage = stage
        self.reason = reason
        self.log = log  # full details, shown with --verbose

    def to_json(self):
        return {"status": self.status, "reason": self.reason, "stage": self.stage}


def run_cmd(argv, timeout, cwd=None, env=None, stdin_data=None):
    """Run argv; return (returncode, stdout, stderr, timed_out).

    The child gets its own process group, so on timeout the whole group is
    killed.  Output is decoded as UTF-8 with replacement.  A negative
    returncode means "killed by signal -returncode".
    """
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, start_new_session=True,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        return 127, b"", str(e).encode(), False
    try:
        out, err = proc.communicate(stdin_data, timeout=timeout)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate()
        return proc.returncode, out, err, True


def text(b, limit=200000):
    return b[:limit].decode("utf-8", "replace")


_LOC = re.compile(r"(?:/[^\s:]+|[\w.\-]+\.[ch]):\d+(?::\d+)?:\s*")


def short(s, n=160):
    # Drop file:line:col prefixes so long absolute paths don't eat the message.
    s = _LOC.sub("", " ".join(s.split()))
    return s if len(s) <= n else s[: n - 3] + "..."


def signame(rc):
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return "signal %d" % -rc


def is_real_cc(cc_argv):
    """True if --cc is a real compiler (gcc/clang), i.e. not hcc."""
    name = os.path.basename(cc_argv[0])
    return bool(re.match(r"^(gcc|cc|clang|tcc)(-[\d.]+)?$", name))


def cc_env():
    """Environment for running the compiler: repo root on PYTHONPATH so that
    'python3 -m hcc' works from any cwd."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    return env


# ------------------------------------------------- compiler-error analysis

TRACEBACK_RE = re.compile(r'File "([^"]*)", line \d+, in ')
STAGE_MARKER_RE = re.compile(
    r"(?:\bstage[=:]\s*|\[)(preprocess|parse|sema|codegen)\b"
    r"|\b(preprocess|parse|sema|codegen)(?:or)? error\b", re.I)

# Map hcc module file names to stages (first matching keyword wins).
MODULE_STAGE = [
    (("preproc", "cpp", "macro", "pp_", "/pp"), "preprocess"),
    (("lex", "token", "pars", "ast"), "parse"),
    (("sema", "type", "check", "resolve", "scope", "const"), "sema"),
    (("codegen", "gen", "emit", "asm", "ir", "reg", "x86", "abi"), "codegen"),
]


def stage_from_stderr(err):
    """Guess the failing compiler stage from stderr, or None."""
    m = STAGE_MARKER_RE.search(err)
    if m:
        return (m.group(1) or m.group(2)).lower()
    frames = [f for f in TRACEBACK_RE.findall(err) if "hcc" in f]
    for f in reversed(frames):  # innermost hcc frame first
        base = f.replace("\\", "/").rsplit("hcc/", 1)[-1].lower()
        for keys, stage in MODULE_STAGE:
            if any(k in base for k in keys):
                return stage
    return None


def is_compiler_crash(rc, err):
    """docs/CONTRACT.md: internal compiler error => exit 70 (+ traceback)."""
    return rc < 0 or rc == 70 or "Traceback (most recent call last)" in err


def first_error_line(err):
    lines = [l for l in err.splitlines() if l.strip()]
    for l in lines:
        if "internal compiler error" in l:
            return l
    if "Traceback (most recent call last)" in err:
        return lines[-1] if lines else "traceback"
    for l in lines:
        if "error" in l.lower():
            return l
    return lines[0] if lines else ""


# ----------------------------------------------------------------- suites

class Suite:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(SUITES_DIR, name)
        cfg = dict(SUITE_DEFAULTS)
        path = os.path.join(self.dir, "suite.json")
        if os.path.exists(path):
            with open(path) as f:
                cfg.update(json.load(f))
        self.cfg = cfg
        self.skips = self._load_skips()
        self.driver = load_driver(cfg["driver"]) if cfg.get("driver") else None

    def subst(self, s):
        return s.replace("{suite}", self.dir).replace("{tests}", TESTS_DIR)

    def _load_skips(self):
        skips = {}
        path = os.path.join(self.dir, self.cfg["skip_file"])
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    body, _, reason = line.partition("#")
                    body = body.strip()
                    if body:
                        skips[body] = reason.strip()
        return skips

    def tests(self):
        """Sorted list of relative paths of test files (not skipped-filtered)."""
        if self.driver is not None and hasattr(self.driver, "list_tests"):
            return sorted(self.driver.list_tests(self))
        import glob as globmod
        found = set()
        for pat in self.cfg["glob"]:
            for p in globmod.glob(os.path.join(self.dir, pat), recursive=True):
                found.add(os.path.relpath(p, self.dir))
        return sorted(found)

    def test_id(self, rel):
        return "%s/%s" % (self.name, rel)

    def cflags(self, cc_argv):
        flags = list(self.cfg["cflags"])
        if is_real_cc(cc_argv):
            flags += self.cfg["gcc_cflags"]
        return [self.subst(f) for f in flags]


DRIVERS_DIR = os.path.join(TESTS_DIR, "drivers")


def load_driver(name):
    """Load tests/drivers/<name>.py.  A driver lets a suite define its own test
    pipeline (e.g. build a whole program with hcc once, then run many scripts).
    It may define list_tests(suite) -> [relpath] and must define
    run_test(suite, rel, cc_argv, work, R) -> R.Result, where R is this module.
    See tests/README.md."""
    import importlib.util
    if DRIVERS_DIR not in sys.path:
        sys.path.insert(0, DRIVERS_DIR)
    spec = importlib.util.spec_from_file_location(
        "hcc_driver_" + name, os.path.join(DRIVERS_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_suites():
    return sorted(d for d in os.listdir(SUITES_DIR)
                  if os.path.isdir(os.path.join(SUITES_DIR, d)))


_support_lock = threading.Lock()
_support_cache = {}


def support_objects(suite, workroot):
    """Compile the suite's support files with host gcc once; return .o paths."""
    with _support_lock:
        if suite.name in _support_cache:
            return _support_cache[suite.name]
        objs = []
        for i, src in enumerate(suite.cfg["support"]):
            src = os.path.join(suite.dir, suite.subst(src))
            obj = os.path.join(workroot, "support-%s-%d.o" % (suite.name, i))
            rc, _, err, _ = run_cmd(["gcc", "-w", "-O0", "-xc", "-c", "-o", obj, src],
                                    TOOL_TIMEOUT)
            if rc != 0:
                raise RuntimeError("support file %s failed to build:\n%s" % (src, text(err)))
            objs.append(obj)
        _support_cache[suite.name] = objs
        return objs


# ------------------------------------------------------------ one test

def read_expectations(src, suite):
    """Return (expected_exit, expected_stdout or None)."""
    exp_exit = suite.cfg["expect_exit"]
    exp_out = None
    if os.path.exists(src + ".exitcode"):
        with open(src + ".exitcode") as f:
            exp_exit = int(f.read().strip())
    if os.path.exists(src + ".expected"):
        with open(src + ".expected", "rb") as f:
            exp_out = f.read()
    return exp_exit, exp_out


def compile_failure(cc_argv, cflags, src, work, rc, err):
    """Classify a failed compile into a Result."""
    err_s = text(err)
    stage = stage_from_stderr(err_s)
    crashed = is_compiler_crash(rc, err_s)
    if stage is None:
        # Disambiguate: does preprocessing alone succeed?
        prc, _, _, _ = run_cmd(cc_argv + cflags + ["-E", "-o", os.devnull, src],
                               COMPILE_TIMEOUT, cwd=work, env=cc_env())
        stage = "preprocess" if prc != 0 else "parse"
    reason = ("signal %s" % signame(rc)) if rc < 0 else first_error_line(err_s)
    log = "compile command: %s\nexit %d\n%s" % (
        shlex.join(cc_argv + cflags + ["-S", "-o", "t.s", src]), rc, err_s)
    return Result("CRASH" if crashed else "FAIL", stage, short(reason), log)


def run_test(suite, rel, cc_argv, workroot, keep):
    src = os.path.join(suite.dir, rel)
    work = tempfile.mkdtemp(prefix="t-", dir=workroot)
    try:
        res = _run_test(suite, src, cc_argv, work)
    except Exception as e:  # infrastructure bug: never kill the whole run
        res = Result("CRASH", None, short("runner error: %r" % e), repr(e))
    if keep:
        dest = os.path.join(keep, suite.test_id(rel).replace("/", "__"))
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(work, dest)
        with open(os.path.join(dest, "log.txt"), "w") as f:
            f.write(res.log)
    shutil.rmtree(work, ignore_errors=True)
    return res


def _run_test(suite, src, cc_argv, work):
    if suite.driver is not None:
        return suite.driver.run_test(suite, os.path.relpath(src, suite.dir), cc_argv,
                                     work, sys.modules[__name__])
    cflags = suite.cflags(cc_argv)
    if os.path.exists(src + ".flags"):  # per-test extra cflags, whitespace separated
        with open(src + ".flags") as f:
            cflags += shlex.split(f.read())
    asm = os.path.join(work, "t.s")
    obj = os.path.join(work, "t.o")
    exe = os.path.join(work, "t")

    # 1. compile to assembly
    rc, _, err, to = run_cmd(cc_argv + cflags + ["-S", "-o", asm, src],
                             COMPILE_TIMEOUT, cwd=work, env=cc_env())
    if to:
        return Result("TIMEOUT", "timeout", "compile >%ds" % COMPILE_TIMEOUT)
    if rc != 0 or not os.path.exists(asm):
        if rc == 0:
            err += b"\n(no assembly output produced)"
            rc = 1
        return compile_failure(cc_argv, cflags, src, work, rc, err)

    # 2. assemble
    rc, _, err, to = run_cmd(["as", "--64", "-o", obj, asm], TOOL_TIMEOUT, cwd=work)
    if to or rc != 0:
        e = text(err)
        lines = [l for l in e.splitlines() if "Error" in l] or e.splitlines() or ["as failed"]
        return Result("FAIL", "assemble", short(lines[0].split(": ", 1)[-1]), e)

    # 3. link
    objs = support_objects(suite, os.path.dirname(work))
    rc, _, err, to = run_cmd(["gcc", "-no-pie", "-o", exe, obj] + objs + suite.cfg["ldflags"],
                             TOOL_TIMEOUT, cwd=work)
    if to or rc != 0:
        e = text(err)
        lines = [l for l in e.splitlines() if "undefined reference" in l or "error" in l.lower()]
        reason = lines[0] if lines else (e.strip().splitlines() or ["link failed"])[0]
        reason = re.sub(r"^.*?: ", "", reason) if "undefined reference" in reason else reason
        return Result("FAIL", "link", short(reason), e)

    # 4. run
    rc, out, err, to = run_cmd([exe], RUN_TIMEOUT, cwd=work)
    log = "exit %s\n--- stdout ---\n%s\n--- stderr ---\n%s" % (rc, text(out), text(err))
    if to:
        return Result("TIMEOUT", "timeout", "run >%ds" % RUN_TIMEOUT, log)
    last = (text(out).strip().splitlines() or text(err).strip().splitlines() or [""])[-1]
    if rc < 0:
        what = "abort()" if -rc == signal.SIGABRT else signame(rc)
        return Result("FAIL", "run", short("killed by %s %s" % (what, last)), log)

    # 5. compare
    exp_exit, exp_out = read_expectations(src, suite)
    if rc != exp_exit:
        return Result("FAIL", "compare",
                      short("exit %d, expected %d %s" % (rc, exp_exit, last)), log)
    if exp_out is not None and out != exp_out:
        return Result("FAIL", "compare", short(describe_diff(exp_out, out)), log)
    return Result("PASS", None, "", log)


def describe_diff(expected, actual):
    e = text(expected).split("\n")
    a = text(actual).split("\n")
    for i in range(max(len(e), len(a))):
        el = e[i] if i < len(e) else "<EOF>"
        al = a[i] if i < len(a) else "<EOF>"
        if el != al:
            return "stdout line %d: expected %r got %r" % (i + 1, el[:60], al[:60])
    return "stdout differs (whitespace/bytes)"


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description="hcc test runner (see tests/README.md)")
    ap.add_argument("--suite", action="append",
                    help="suite name (repeatable, or comma list, or 'all'); default smoke")
    ap.add_argument("--jobs", "-j", type=int, default=4)
    ap.add_argument("--filter", action="append",
                    help="glob on test id, e.g. 'smoke/*struct*' (repeatable)")
    ap.add_argument("--json", help="write {test_id: {status, reason, stage}} here")
    ap.add_argument("--cc", default="python3 -m hcc", help="compiler command (may contain spaces)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print full log of non-passing tests")
    ap.add_argument("--keep", metavar="DIR", help="keep each test's work dir (t.s, t, log.txt) under DIR")
    ap.add_argument("--baseline", metavar="JSON", help="previous --json; report regressions/fixes")
    ap.add_argument("--list", action="store_true", help="list test ids and exit")
    ap.add_argument("--include-skipped", action="store_true", help="also run tests in skip.txt")
    ap.add_argument("--quiet", "-q", action="store_true", help="only print non-PASS lines")
    args = ap.parse_args(argv)

    names = []
    for s in args.suite or ["smoke"]:
        names += [x for x in s.split(",") if x]
    if "all" in names:
        names = list_suites()
    suites = []
    for n in names:
        if not os.path.isdir(os.path.join(SUITES_DIR, n)):
            ap.error("unknown suite %r; available: %s" % (n, ", ".join(list_suites())))
        suites.append(Suite(n))

    jobs, skipped = [], 0
    for suite in suites:
        for rel in suite.tests():
            tid = suite.test_id(rel)
            if args.filter and not any(fnmatch.fnmatch(tid, f) for f in args.filter):
                continue
            if rel in suite.skips and not args.include_skipped:
                skipped += 1
                continue
            jobs.append((suite, rel, tid))

    if args.list:
        for _, _, tid in jobs:
            print(tid)
        return 0

    cc_argv = shlex.split(args.cc)
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
        args.keep = os.path.abspath(args.keep)
    workroot = tempfile.mkdtemp(prefix="hcc-tests-")
    results = {}
    t0 = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as ex:
            futs = {ex.submit(run_test, s, rel, cc_argv, workroot, args.keep): tid
                    for s, rel, tid in jobs}
            for fut in concurrent.futures.as_completed(futs):
                tid = futs[fut]
                res = fut.result()
                results[tid] = res
                if not (args.quiet and res.status == "PASS"):
                    print(("%s %s %s" % (res.status, tid, res.reason)).rstrip(), flush=True)
                if args.verbose and res.status != "PASS":
                    print("    " + res.log.replace("\n", "\n    ")[:4000], flush=True)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    counts = {k: 0 for k in ("PASS", "FAIL", "TIMEOUT", "CRASH")}
    by_stage = {}
    for r in results.values():
        counts[r.status] += 1
        if r.status != "PASS":
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: results[k].to_json() for k in sorted(results)}, f, indent=1, sort_keys=True)
    if args.baseline:
        report_baseline(args.baseline, results)
    stages = " ".join("%s=%d" % (k, by_stage[k]) for k in STAGES if k in by_stage)
    print("SUMMARY suites=%s total=%d pass=%d fail=%d timeout=%d crash=%d skipped=%d time=%.1fs%s"
          % (",".join(names), len(results), counts["PASS"], counts["FAIL"], counts["TIMEOUT"],
             counts["CRASH"], skipped, time.time() - t0, (" | " + stages) if stages else ""))
    return 0 if counts["PASS"] == len(results) else 1


def report_baseline(path, results):
    with open(path) as f:
        old = json.load(f)
    regress = sorted(t for t, r in results.items()
                     if r.status != "PASS" and old.get(t, {}).get("status") == "PASS")
    fixed = sorted(t for t, r in results.items()
                   if r.status == "PASS" and t in old and old[t]["status"] != "PASS")
    for t in regress:
        print("REGRESSION %s %s" % (t, results[t].reason))
    for t in fixed:
        print("FIXED %s" % t)
    print("BASELINE regressions=%d fixed=%d" % (len(regress), len(fixed)))


if __name__ == "__main__":
    sys.exit(main())
