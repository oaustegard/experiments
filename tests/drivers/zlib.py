"""Driver for the `zlib` suite: zlib 1.3.1 built by hcc.

Test ids:
  zlib/src/<f>.c        hcc compiles that one library file; the object is linked with the gcc -O0
                        objects of the rest into `example` and `minigzip` (upstream test programs),
                        and both must behave as in an all-gcc build: example's output, a gzip
                        round trip of a fixed text, and minigzip's compressed bytes.
  zlib/test/<prog>.c    upstream `make test` programs, entirely built by hcc: example (self test,
                        writes a gz file), minigzip (`echo hello world | minigzip | minigzip -d`),
                        infcover (inflate coverage test using internal headers).
  zlib/gz/<input>-<mode>  entirely hcc-built minigzip compresses <input> with <mode>; the bytes
                        must equal the gcc-built minigzip's, and the hcc build must inflate them
                        back to the input.  Inputs are zlib's own files plus generated corpora.
"""
import hashlib
import os

import common as C
import projlib as P

URL = "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz"
SHA256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"
LIB = ["adler32", "compress", "crc32", "deflate", "gzclose", "gzlib", "gzread", "gzwrite",
       "infback", "inffast", "inflate", "inftrees", "trees", "uncompr", "zutil"]
INPUTS = ["zlib.h", "deflate.c", "trees.c", "ChangeLog", "zeros", "random", "runs", "words", "tiny"]
MODES = {"1": ["-1"], "6": ["-6"], "9": ["-9"], "f": ["-f"], "h": ["-h"], "r": ["-r"], "9f": ["-9", "-f"]}
RUN_TIMEOUT = 60


def root():
    return P.fetch_tarball("zlib", URL, SHA256)


def list_tests(suite):
    out = ["src/%s.c" % u for u in LIB] + ["test/example.c", "test/minigzip.c", "test/infcover.c"]
    out += ["gz/%s-%s" % (i, m) for i in INPUTS for m in MODES]
    return out


def project(R, suite, cc_argv):
    r = root()
    units = {u: os.path.join(r, u + ".c") for u in LIB}
    for t in ("example", "minigzip", "infcover"):
        units[t] = os.path.join(r, "test", t + ".c")
    return P.Project(R, suite, cc_argv, "zlib", units, ["-I" + r, "-D_LARGEFILE64_SOURCE=1"])


def corpus(r, name):
    if name == "zeros":
        return bytes(200000)
    if name == "tiny":
        return b"hello world\n"
    if name == "random":
        x, out = 12345, bytearray()
        for _ in range(65536):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x >> 16) & 0xFF)
        return bytes(out)
    if name == "runs":
        out = bytearray()
        for i in range(400):
            out += bytes([65 + i % 26]) * (1 + (i * 7) % 90) + b"abcabcabc" * (i % 5)
        return bytes(out)
    if name == "words":
        w = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]
        x, out = 7, []
        for _ in range(12000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(w[(x >> 16) % len(w)])
        return (" ".join(out) + "\n").encode()
    with open(os.path.join(r, name), "rb") as f:
        return f.read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def runb(R, argv, cwd, stdin_data=None, timeout=RUN_TIMEOUT):
    rc, out, err, to = P.run_cmd(R, argv, timeout, cwd=cwd, stdin_data=stdin_data)
    return rc, out, R.text(err), to


def example_run(R, exe, work):
    d = P.tmpdir(work, "ex")
    return P.run(R, [exe, os.path.join(d, "tmp.gz")], d, RUN_TIMEOUT)


def smoke_mgz(R, exe, work, data):
    d = P.tmpdir(work, "mg")
    rc, gz, err, to = runb(R, [exe, "-6"], d, data)
    rc2, back, err2, to2 = runb(R, [exe, "-d"], d, gz)
    return {"rc": rc or rc2, "gz": sha(gz), "ok": back == data, "timeout": to or to2}


def run_test(suite, rel, cc_argv, work, R):
    proj = project(R, suite, cc_argv)
    r = root()
    if rel.startswith("src/"):
        return run_src(R, proj, rel[4:-2], work, r)
    if rel.startswith("test/"):
        return run_prog(R, proj, rel[5:-2], work, r)
    return run_gz(R, proj, rel[3:], work, r)


def link_pair(R, proj, hcc_units, work):
    exes = {}
    for t in ("example", "minigzip"):
        out = os.path.join(work, t)
        f = proj.mixed(out, LIB + [t], hcc_units, ["-lm"])
        if f:
            return None, f
        exes[t] = out
    return exes, None


def smoke(R, exes, work, data):
    return example_run(R, exes["example"], work), smoke_mgz(R, exes["minigzip"], work, data)


def run_src(R, proj, unit, work, r):
    exes, fail = link_pair(R, proj, [unit], work)
    if fail:
        return fail.result(R)
    data = corpus(r, "words")
    ref = {t: proj.ref(t, LIB + [t], ["-lm"]) for t in ("example", "minigzip")}
    want = P.ref_value("zlib-smoke", [C.file_sha(ref["example"]), C.file_sha(ref["minigzip"]), C.file_sha(__file__)],
                       lambda: dict(zip(("ex", "mg"), smoke(R, ref, P.tmpdir(work, "ref"), data))))
    ex, mg = smoke(R, exes, P.tmpdir(work, "got"), data)
    res = C.compare_runs(R, ex, want["ex"])
    if res.status != "PASS":
        return res
    if mg != want["mg"]:
        return R.Result("FAIL", "compare", R.short("minigzip output differs: got %s expected %s" % (mg, want["mg"])))
    return R.Result("PASS")


def run_prog(R, proj, prog, work, r):
    exe, fail = proj.full(prog, LIB + [prog], ["-lm"])
    if fail:
        return fail.result(R)
    ref = proj.ref(prog, LIB + [prog], ["-lm"])
    if prog == "example":
        got = example_run(R, exe, P.tmpdir(work, "got"))
        want = P.ref_value("zlib-example", [C.file_sha(ref)], lambda: example_run(R, ref, P.tmpdir(work, "ref")))
        return C.compare_runs(R, got, want)
    if prog == "minigzip":
        data = b"hello world\n"
        got = smoke_mgz(R, exe, P.tmpdir(work, "got"), data)
        want = P.ref_value("zlib-mgz", [C.file_sha(ref)], lambda: smoke_mgz(R, ref, P.tmpdir(work, "ref"), data))
        if got != want or not got["ok"]:
            return R.Result("FAIL", "compare", R.short("echo hello world | minigzip | minigzip -d: %s (expected %s)" % (got, want)))
        return R.Result("PASS")
    got = P.run(R, [exe], P.tmpdir(work, "got"), 120)
    want = P.ref_value("zlib-infcover", [C.file_sha(ref)], lambda: P.run(R, [ref], P.tmpdir(work, "ref"), 120))
    return C.compare_runs(R, got, want)


def run_gz(R, proj, tid, work, r):
    name, mode = tid.rsplit("-", 1)
    data = corpus(r, name)
    exe, fail = proj.full("minigzip", LIB + ["minigzip"], ["-lm"])
    if fail:
        return fail.result(R)
    ref = proj.ref("minigzip", LIB + ["minigzip"], ["-lm"])
    d = P.tmpdir(work, "gz")
    want = P.ref_value("zlib-gz", [C.file_sha(ref), sha(data), mode],
                       lambda: sha(runb(R, [ref] + MODES[mode], d, data)[1]))
    rc, gz, err, to = runb(R, [exe] + MODES[mode], d, data)
    if to:
        return R.Result("TIMEOUT", "timeout", "compress >%ds" % RUN_TIMEOUT)
    if rc != 0:
        return R.Result("FAIL", "run", R.short("minigzip exit %d %s" % (rc, err)))
    if sha(gz) != want:
        return R.Result("FAIL", "compare", "compressed output differs from gcc build (%d bytes)" % len(gz))
    rc, back, err, to = runb(R, [exe, "-d"], d, gz)
    if rc != 0 or back != data:
        return R.Result("FAIL", "compare", R.short("round trip differs (exit %d, %d bytes of %d) %s" % (rc, len(back), len(data), err)))
    return R.Result("PASS")
