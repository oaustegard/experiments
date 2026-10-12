#!/usr/bin/env python3
"""Regenerate tests/suites/mbedtls/manifest.json: build every Mbed TLS test suite with gcc -O0, run it, and list the
test cases that actually run (status PASS; cases skipped for unmet dependencies are left out).  Suites with more
than 600 live cases are sampled with a fixed stride (the manifest records it).  Needs network on first use
(tarball into tests/tools) and takes a couple of minutes.

    python3 tests/scripts/gen_mbedtls_manifest.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import run as R  # noqa: E402


def main():
    suite = R.Suite("mbedtls")
    m = suite.driver.build_manifest(R, suite)
    path = os.path.join(suite.dir, "manifest.json")
    with open(path, "w") as f:
        json.dump({"lib": m["lib"], "helpers": m["helpers"],
                   "suites": {s: {"total": i["total"], "live": i["live"], "stride": i["stride"], "cases": i["cases"]}
                              for s, i in sorted(m["suites"].items())}}, f, separators=(",", ":"))
        f.write("\n")
    live = sum(len(i["cases"]) for i in m["suites"].values())
    print("%d suites, %d listed cases (of %d live), written to %s" % (len(m["suites"]), live, sum(i["live"] for i in m["suites"].values()), path))


if __name__ == "__main__":
    main()
