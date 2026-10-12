#!/usr/bin/env python3
"""Regenerate tests/suites/cpython/manifest.json: build CPython with gcc (once, into tests/cache), run each candidate
regression-test module twice with `python -m test -v`, and list the unittest methods that pass in both runs (at most
400 per module, by stride).  Takes about ten minutes the first time.

    python3 tests/scripts/gen_cpython_manifest.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import run as R  # noqa: E402


def main():
    suite = R.Suite("cpython")
    m = suite.driver.build_manifest(R, suite)
    path = os.path.join(suite.dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"))
        f.write("\n")
    print("%d units, %d modules, %d listed cases (of %d live)" % (
        len(m["units"]), len(m["modules"]), sum(len(i["cases"]) for i in m["modules"].values()),
        sum(i["live"] for i in m["modules"].values())))


if __name__ == "__main__":
    main()
