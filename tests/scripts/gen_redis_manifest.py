#!/usr/bin/env python3
"""Regenerate tests/suites/redis/manifest.json: build Redis with gcc (once, into tests/cache), run each candidate
Tcl test file twice with `./runtest --single`, and list the test cases that pass in both runs (at most
400 per module, by stride).  Takes a long time the first time (gcc build of redis, then every candidate file twice).

    python3 tests/scripts/gen_redis_manifest.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import run as R  # noqa: E402


def main():
    suite = R.Suite("redis")
    m = suite.driver.build_manifest(R, suite)
    path = os.path.join(suite.dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"))
        f.write("\n")
    print("%d units, %d files, %d listed cases (of %d live)" % (
        len(m["units"]), len(m["files"]), sum(len(i["cases"]) for i in m["files"].values()),
        sum(i["live"] for i in m["files"].values())))


if __name__ == "__main__":
    main()
