#!/usr/bin/env python3
"""Scan sqlite-src/test/*.test with a gcc-built testfixture and record which files run clean and fast
(rc 0, "0 errors out of N tests", under --limit seconds).  Output: tests/cache/sqlite_tcl_scan.json
and (with --write) tests/suites/sqlite/tcl-tests.txt, the manifest of the `sqlite/tcl/*` tests.

    python3 tests/scripts/scan_sqlite_tcl.py --testfixture /path/to/testfixture --jobs 2 [--limit 20] [--write]
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
SRC = os.path.join(TESTS, "tools", "sqlite-srcdist", "sqlite-src-3490100", "test")
SKIP = re.compile(r"^(all|veryquick|quick|soak|releasetest|permutations|fuzz\d*|fuzzer.*|speed.*|malloc.*|oom.*|crash.*|"
                  r"journal.*|walcrash.*|walthread|thread.*|mt.*|backup_ioerr|ioerr.*|shell.*|corrupt.*|savepoint7|"
                  r"fts3.*|fts4.*|lock.*|multiplex.*|sqllimits1|bigfile.*|trace.*|pager.*|tkt.*|collate.*)$", re.I)


def run_one(tf, name, limit):
    work = tempfile.mkdtemp(prefix="tclscan-")
    t0 = time.time()
    try:
        r = subprocess.run([tf, os.path.join(SRC, name + ".test")], cwd=work, capture_output=True, text=True, timeout=limit,
                           errors="replace", env=dict(os.environ, TMPDIR=work))
        out = r.stdout + r.stderr
        m = re.findall(r"(\d+) errors out of (\d+) tests", out)
        return {"rc": r.returncode, "secs": time.time() - t0, "summary": m[-1] if m else None}
    except subprocess.TimeoutExpired:
        return {"rc": None, "secs": limit, "summary": None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testfixture", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--write", action="store_true", help="write the manifest (files that run clean in <= --manifest-secs)")
    ap.add_argument("--manifest-secs", type=float, default=5.0)
    args = ap.parse_args()
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(SRC, "*.test")))
    names = [n for n in names if not SKIP.match(n)]
    res = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(run_one, os.path.abspath(args.testfixture), n, args.limit): n for n in names}
        for f in concurrent.futures.as_completed(futs):
            res[futs[f]] = f.result()
            r = res[futs[f]]
            print(futs[f], r["rc"], "%.1fs" % r["secs"], r["summary"], flush=True)
    os.makedirs(os.path.join(TESTS, "cache"), exist_ok=True)
    with open(os.path.join(TESTS, "cache", "sqlite_tcl_scan.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    good = sorted(n for n, r in res.items() if r["rc"] == 0 and r["summary"] and r["summary"][0] == "0" and int(r["summary"][1]) > 0 and r["secs"] <= args.manifest_secs)
    print("%d of %d run clean within %ds" % (len(good), len(names), args.limit))
    if args.write:
        with open(os.path.join(TESTS, "suites", "sqlite", "tcl-tests.txt"), "w") as f:
            f.write("\n".join(good) + "\n")


if __name__ == "__main__":
    main()
