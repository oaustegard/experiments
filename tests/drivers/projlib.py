"""Shared machinery for the real-project suites (zlib, bzip2, cjson, tcc, chibicc-self,
doom, mbedtls, redis, cpython).  See tests/README.md.

A project is a set of C translation units ("units": name -> source path) and one or more
programs, each a list of units.  Everything is cached on disk by tests/drivers/common.py
(per hcc source hash for hcc-built objects, per gcc version for the references):

  * `Project.hobj(name)`   unit compiled by the compiler under test (cached; a failure is cached too)
  * `Project.gobj(name)`   unit compiled by gcc -O0
  * `Project.mixed(...)`   a program whose chosen units come from hcc and the rest from gcc
                           (the "compiles + links into a gcc-built remainder" test)
  * `Project.full(...)`    a program whose units all come from hcc (cached)
  * `Project.ref(...)`     the same program all-gcc (cached): the reference

Sources are fetched into tests/tools (gitignored) and pinned by sha256 (tarballs) or commit
(git).  Only compiling is done by hcc; assembling, linking and running use the host tools.
"""
import concurrent.futures
import os
import shutil
import subprocess
import tarfile
import threading
import zipfile

import common as C


# ----------------------------------------------------------------- fetching

def fetch_tarball(name, url, sha256, subdir=None):
    """Download (once) and extract into tests/tools/ex/<name>; returns the top directory.
    `subdir` is the directory inside the archive to return (default: the only top-level dir)."""
    root = os.path.join(C.tools_root(), "ex", name)
    marker = os.path.join(root, ".extracted")
    if not os.path.exists(marker):
        dl = C.fetch(url, os.path.join(C.tools_root(), "dl", os.path.basename(url)), sha256)
        with C.locked(root):
            if not os.path.exists(marker):
                os.makedirs(root, exist_ok=True)
                if dl.endswith(".zip"):
                    with zipfile.ZipFile(dl) as z:
                        z.extractall(root)
                else:
                    with tarfile.open(dl) as t:
                        t.extractall(root, filter="data")
                open(marker, "w").close()
    if subdir:
        return os.path.join(root, subdir)
    tops = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    return os.path.join(root, tops[0])


def fetch_git(name, url, ref, commit):
    """Fetch exactly `commit` of url (shallow) into tests/tools/src/<name> once.  `ref` is only documentation
    (the tag or branch the commit came from)."""
    d = os.path.join(C.tools_root(), "src", name)
    with C.locked(d):
        head = None
        if os.path.isdir(os.path.join(d, ".git")):
            head = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != commit:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
            for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", url],
                        ["git", "fetch", "-q", "--depth", "1", "origin", commit], ["git", "checkout", "-q", "FETCH_HEAD"]):
                r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError("%s failed: %s" % (" ".join(cmd), r.stderr[-500:]))
    return d


def copy_tree(src, dst):
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)


def tmpdir(work, name):
    d = os.path.join(work, name)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ project

# At most this many hcc compiles run at once inside one test process, however many tests wait for objects
# (the swarm box is shared: $HCC_TEST_COMPILE_JOBS, default 2).
_COMPILE_SEM = threading.Semaphore(int(os.environ.get("HCC_TEST_COMPILE_JOBS", "2")))

class Project:
    """units: {name: source path}.  flags: cflags for hcc and gcc (hcc additionally gets the
    suite's cflags for a real cc via suite.cflags).  unit_flags: {name: extra flags}."""

    def __init__(self, R, suite, cc_argv, tag, units, flags, ldflags=(), unit_flags=None, key=""):
        self.R, self.suite, self.cc_argv, self.tag = R, suite, cc_argv, tag
        self.units, self.flags, self.ldflags = dict(units), list(flags), list(ldflags)
        self.unit_flags = unit_flags or {}
        self.key = key
        self.extra_objs = []  # objects linked into every program (e.g. assembled .S files)
        self.hflags = list(suite.cflags(cc_argv)) + self.flags
        self.gflags = list(suite.cfg["cflags"]) + self.flags

    # -- objects
    def _extra(self, name):
        return list(self.unit_flags.get(name, []))

    def hobj(self, name, timeout=900):
        with _COMPILE_SEM:
            return C.cached_object(self.R, self.cc_argv, self.hflags + self._extra(name), self.units[name],
                                   "%s/%s" % (self.tag, name), timeout, extra_key=self.key)

    def gobj(self, name):
        return C.gcc_object(self.gflags + self._extra(name), self.units[name], "%s/%s" % (self.tag, name))

    # -- programs
    def _link(self, out, names, hcc_names, ldflags, extra_objs=(), degrade=False):
        objs, first = [], None
        for n in names:
            if n in hcc_names:
                o, fail = self.hobj(n)
                if fail and degrade:
                    objs.append(self.gobj(n))  # unit hcc cannot compile yet: the gcc object stands in
                elif fail:
                    first = first or C.Failure(fail.status, fail.stage, "%s build: %s: %s" % (self.tag, n, fail.reason), fail.log)
                else:
                    objs.append(o)
            else:
                objs.append(self.gobj(n))
        if first:
            return first
        return C.link(self.R, out, objs + list(extra_objs) + list(self.extra_objs), list(ldflags) + self.ldflags, os.path.dirname(out) or ".")

    def mixed(self, out, names, hcc_names, ldflags=(), extra_objs=()):
        """Link `names` into out; units in hcc_names come from hcc.  Returns None or a Failure."""
        return self._link(out, names, set(hcc_names), ldflags, extra_objs)

    def _key(self, label, names):
        srcs = [C.file_sha(self.units[n]) for n in names]
        return [label, " ".join(self.flags), self.key, *srcs, *[" ".join(self._extra(n)) for n in names]]

    def full(self, label, names, ldflags=(), jobs=2, degrade=False):
        """Cached program with every unit built by the compiler under test.  With degrade=True a unit that fails
        to compile is replaced by its gcc object, so the program exists whenever it links (the failing unit has its
        own test); otherwise the first failure is returned."""
        key = [C.compiler_fingerprint(self.R, self.cc_argv)] + self._key(label, names) + [" ".join(self.hflags), str(degrade)]

        def build(out):
            with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
                list(ex.map(self.hobj, names))  # warm the object cache in parallel; failures are cached too
            return self._link(out, names, set(names), ldflags, degrade=degrade)
        return C.cached_program(self.R, self.tag + "-hcc", key, build)

    def ref(self, label, names, ldflags=()):
        """Cached all-gcc program (the reference)."""
        key = [C.gcc_fingerprint()] + self._key(label, names)

        def build(out):
            return self._link(out, names, set(), ldflags)
        path, fail = C.cached_program(self.R, self.tag + "-ref", key, build)
        if fail:
            raise RuntimeError("reference %s %s failed to link: %s" % (self.tag, label, fail.reason))
        return path


def ref_value(name, parts, compute):
    """Cache a reference result (JSON-serialisable) keyed on the gcc version and `parts`."""
    return C.cached_value(name, [C.gcc_fingerprint()] + list(parts), compute)


def run(R, argv, cwd, timeout=60, stdin_data=None, env=None):
    return C.run_and_capture(R, argv, timeout, cwd, env=env, stdin_data=stdin_data)
