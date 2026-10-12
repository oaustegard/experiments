#!/usr/bin/env python3
"""Bound a big suite's run time by sampling: keep every test the compiler under test currently fails and a
deterministic subset of the ones it passes; the rest go to skip.txt as "sampled out".

    python3 tests/scripts/sample_suite.py gcc-dg-compile RESULTS.json --total 3300

RESULTS.json is `tests/run.py --suite NAME --json RESULTS.json` for the full suite.  Passing tests are ranked by
sha256(test id) and the first (total - failing) are kept, so the choice does not depend on file order or on the
machine.  Earlier "sampled out" entries are dropped first, so the script can be re-run with a new RESULTS.json (a
test the compiler now fails comes back; the previously failing ones that now pass stay as long as they rank in).
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = os.path.join(os.path.dirname(HERE), "suites")
MARK = "sampled out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("results")
    ap.add_argument("--total", type=int, required=True, help="active tests to keep (all failing ones plus passing ones)")
    a = ap.parse_args()
    skip_path = os.path.join(SUITES, a.suite, "skip.txt")
    lines = [l for l in open(skip_path).read().splitlines() if MARK not in l]
    res = json.load(open(a.results))
    failing = sorted(t for t, r in res.items() if r["status"] != "PASS")
    passing = sorted((t for t, r in res.items() if r["status"] == "PASS"),
                     key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    keep = max(0, a.total - len(failing))
    drop = passing[keep:]
    with open(skip_path, "w") as f:
        for l in lines:
            f.write(l + "\n")
        for t in sorted(drop):
            f.write("%s  # %s: passes with hcc; deterministic sample (tests/scripts/sample_suite.py) bounds the run time\n"
                    % (t.split("/", 1)[1], MARK))
    print("%s: %d failing kept, %d of %d passing kept, %d sampled out" % (a.suite, len(failing), min(keep, len(passing)), len(passing), len(drop)))


if __name__ == "__main__":
    main()
