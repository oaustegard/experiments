"""Driver for the `bzip2` suite: bzip2 1.0.8 built by hcc.

Test ids:
  bzip2/src/<f>.c         hcc compiles that one file (the 7 library files, bzip2.c, bzip2recover.c);
                          it is linked with gcc -O0 objects of the rest.  The upstream `make test`
                          must hold: -1/-2/-3 compression of sample{1,2,3}.ref gives the shipped
                          .bz2 bytes and the .bz2 files decompress (-d, -ds) to the .ref files.
                          bzip2recover.c: recovering sample1.bz2 gives the gcc build's files.
  bzip2/make-test/<n>     the same upstream `make test` steps with an entirely hcc-built bzip2
                          (n = sample1..3 compress, decompress, small-mode decompress).
  bzip2/level/<in>-<l>    hcc-built bzip2 -<l> compresses <in>: bytes must equal the gcc build's and
                          decompress back to <in>.
  bzip2/mode/<m>          other command-line modes on files: -t, -c, -k with file names, -d -c.
"""
import hashlib
import os

import common as C
import projlib as P

URL = "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"
SHA256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269"
LIB = ["blocksort", "bzlib", "compress", "crctable", "decompress", "huffman", "randtable"]
INPUTS = ["sample1.ref", "sample2.ref", "sample3.ref", "words", "random", "zeros", "runs"]
LEVELS = ["1", "2", "5", "9"]
MODES = ["test", "stdout-file", "keep-file", "decomp-stdout", "small-compress", "verbose"]
RUN_TIMEOUT = 120


def root():
    return P.fetch_tarball("bzip2", URL, SHA256)


def list_tests(suite):
    out = ["src/%s.c" % u for u in LIB + ["bzip2", "bzip2recover"]]
    out += ["make-test/%s" % n for n in ("sample1", "sample2", "sample3", "decompress", "decompress-small")]
    out += ["level/%s-%s" % (i, l) for i in INPUTS for l in LEVELS]
    out += ["mode/%s" % m for m in MODES]
    return out


def project(R, suite, cc_argv):
    r = root()
    units = {u: os.path.join(r, u + ".c") for u in LIB + ["bzip2", "bzip2recover"]}
    return P.Project(R, suite, cc_argv, "bzip2", units, ["-I" + r, "-D_FILE_OFFSET_BITS=64"])


def sha(b):
    return hashlib.sha256(b).hexdigest()


def corpus(r, name):
    if name == "zeros":
        return bytes(300000)
    if name == "random":
        x, out = 99, bytearray()
        for _ in range(100000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x >> 16) & 0xFF)
        return bytes(out)
    if name == "runs":
        out = bytearray()
        for i in range(600):
            out += bytes([48 + i % 70]) * (1 + (i * 13) % 300)
        return bytes(out)
    if name == "words":
        w = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "bzip2", "block", "sorting"]
        x, out = 3, []
        for _ in range(30000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append(w[(x >> 16) % len(w)])
        return (" ".join(out) + "\n").encode()
    with open(os.path.join(r, name), "rb") as f:
        return f.read()


def runb(R, argv, cwd, stdin_data=None, timeout=RUN_TIMEOUT):
    rc, out, err, to = P.run_cmd(R, argv, timeout, cwd=cwd, stdin_data=stdin_data)
    return rc, out, R.text(err), to


def make_test(R, exe, r, work):
    """The upstream `make test` steps; returns a JSON-able summary of every output."""
    d = P.tmpdir(work, "mt")
    res = {}
    for n, lvl in ((1, "-1"), (2, "-2"), (3, "-3")):
        ref = open(os.path.join(r, "sample%d.ref" % n), "rb").read()
        rc, out, err, to = runb(R, [exe, lvl], d, ref)
        res["c%d" % n] = [rc, sha(out), to]
    for n, flag in ((1, "-d"), (2, "-d"), (3, "-ds")):
        bz = open(os.path.join(r, "sample%d.bz2" % n), "rb").read()
        rc, out, err, to = runb(R, [exe, flag], d, bz)
        res["d%d" % n] = [rc, sha(out), to]
    return res


def expected_make_test(r):
    res = {}
    for n in (1, 2, 3):
        res["c%d" % n] = [0, sha(open(os.path.join(r, "sample%d.bz2" % n), "rb").read()), False]
        res["d%d" % n] = [0, sha(open(os.path.join(r, "sample%d.ref" % n), "rb").read()), False]
    return res


def recover(R, exe, r, work):
    d = P.tmpdir(work, "rec")
    bz = open(os.path.join(r, "sample1.bz2"), "rb").read()
    with open(os.path.join(d, "bad.bz2"), "wb") as f:
        f.write(bz[:2000] + b"\x00\xff\x00" + bz[2003:])
    rc, out, err, to = runb(R, [exe, "bad.bz2"], d)
    files = {fn: sha(open(os.path.join(d, fn), "rb").read()) for fn in sorted(os.listdir(d)) if fn.startswith("rec")}
    return {"rc": rc, "files": files, "timeout": to}


def check_make_test(R, got, r):
    want = expected_make_test(r)
    for k in sorted(want):
        if got[k] != want[k]:
            what = "compress" if k[0] == "c" else "decompress"
            return R.Result("FAIL", "compare", "make test: sample%s %s: got %s expected %s" % (k[1], what, got[k][:2], want[k][:2]))
    return R.Result("PASS")


def run_test(suite, rel, cc_argv, work, R):
    proj = project(R, suite, cc_argv)
    r = root()
    kind, _, name = rel.partition("/")
    if kind == "src":
        return run_src(R, proj, name[:-2], work, r)
    if kind == "make-test":
        return run_make(R, proj, name, work, r)
    if kind == "level":
        return run_level(R, proj, name, work, r)
    return run_mode(R, proj, name, work, r)


def prog_units(prog):
    return LIB + [prog]


def run_src(R, proj, unit, work, r):
    if unit == "bzip2recover":
        out = os.path.join(work, "bzip2recover")
        f = proj.mixed(out, ["bzip2recover"], ["bzip2recover"])
        if f:
            return f.result(R)
        ref = proj.ref("bzip2recover", ["bzip2recover"])
        want = P.ref_value("bzip2-recover", [C.file_sha(ref), C.file_sha(__file__)], lambda: recover(R, ref, r, P.tmpdir(work, "ref")))
        got = recover(R, out, r, P.tmpdir(work, "got"))
        return R.Result("PASS") if got == want else R.Result("FAIL", "compare", R.short("bzip2recover: got %s expected %s" % (got, want)))
    out = os.path.join(work, "bzip2")
    f = proj.mixed(out, prog_units("bzip2"), [unit])
    if f:
        return f.result(R)
    return check_make_test(R, make_test(R, out, r, work), r)


def run_make(R, proj, name, work, r):
    exe, fail = proj.full("bzip2", prog_units("bzip2"))
    if fail:
        return fail.result(R)
    got = make_test(R, exe, r, work)
    want = expected_make_test(r)
    sel = {"sample1": ["c1"], "sample2": ["c2"], "sample3": ["c3"],
           "decompress": ["d1", "d2"], "decompress-small": ["d3"]}[name]
    for k in sel:
        if got[k] != want[k]:
            return R.Result("FAIL", "compare", R.short("%s: got %s expected %s" % (k, got[k][:2], want[k][:2])))
    return R.Result("PASS")


def run_level(R, proj, tid, work, r):
    name, lvl = tid.rsplit("-", 1)
    data = corpus(r, name)
    exe, fail = proj.full("bzip2", prog_units("bzip2"))
    if fail:
        return fail.result(R)
    ref = proj.ref("bzip2", prog_units("bzip2"))
    d = P.tmpdir(work, "lv")
    want = P.ref_value("bzip2-level", [C.file_sha(ref), sha(data), lvl], lambda: sha(runb(R, [ref, "-" + lvl], d, data)[1]))
    rc, bz, err, to = runb(R, [exe, "-" + lvl], d, data)
    if to:
        return R.Result("TIMEOUT", "timeout", "compress >%ds" % RUN_TIMEOUT)
    if rc != 0:
        return R.Result("FAIL", "run", R.short("bzip2 exit %d %s" % (rc, err)))
    if sha(bz) != want:
        return R.Result("FAIL", "compare", "compressed output differs from gcc build (%d bytes)" % len(bz))
    rc, back, err, to = runb(R, [exe, "-d"], d, bz)
    if rc != 0 or back != data:
        return R.Result("FAIL", "compare", R.short("round trip differs (exit %d, %d of %d bytes) %s" % (rc, len(back), len(data), err)))
    return R.Result("PASS")


def mode_run(R, exe, r, work, m):
    d = P.tmpdir(work, "m")
    data = corpus(r, "sample1.ref")
    with open(os.path.join(d, "f.txt"), "wb") as fh:
        fh.write(data)
    bz = open(os.path.join(r, "sample2.bz2"), "rb").read()
    with open(os.path.join(d, "g.bz2"), "wb") as fh:
        fh.write(bz)
    res = {}
    if m == "test":
        argv = [exe, "-t", "g.bz2"]
    elif m == "stdout-file":
        argv = [exe, "-c", "-9", "f.txt"]
    elif m == "keep-file":
        argv = [exe, "-k", "-1", "f.txt"]
    elif m == "decomp-stdout":
        argv = [exe, "-d", "-c", "g.bz2"]
    elif m == "small-compress":
        argv = [exe, "-s", "-c", "-2", "f.txt"]
    else:
        argv = [exe, "-vv", "-c", "-3", "f.txt"]
    rc, out, err, to = runb(R, argv, d)
    res["rc"], res["out"], res["timeout"] = rc, sha(out), to
    res["err"] = R.short(err.replace(d, "D"), 300) if m != "verbose" else ""
    res["files"] = {fn: sha(open(os.path.join(d, fn), "rb").read()) for fn in sorted(os.listdir(d))}
    return res


def run_mode(R, proj, m, work, r):
    exe, fail = proj.full("bzip2", prog_units("bzip2"))
    if fail:
        return fail.result(R)
    ref = proj.ref("bzip2", prog_units("bzip2"))
    want = P.ref_value("bzip2-mode", [C.file_sha(ref), m, C.file_sha(__file__)], lambda: mode_run(R, ref, r, P.tmpdir(work, "ref"), m))
    got = mode_run(R, exe, r, P.tmpdir(work, "got"), m)
    if got != want:
        diff = [k for k in want if want[k] != got.get(k)]
        return R.Result("FAIL", "compare", R.short("mode %s differs in %s: got %s expected %s" % (m, diff, [got.get(k) for k in diff], [want[k] for k in diff])))
    return R.Result("PASS")
