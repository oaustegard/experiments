"""Run CPython's own test_difflib (v3.13.16) with difflib.SequenceMatcher
replaced by the candidate's. Everything in difflib that builds a matcher
(Differ, unified_diff, ndiff, HtmlDiff, get_close_matches, the module
doctests) looks the class up as a module global, so the patch reaches all of
it.

test_difflib.py is CPython code, so run this inside claude-workspace's
scripts/jail.sh (see harness/grade.py).

    python3 run_upstream.py --pkg DIR

Prints one JSON line: tests run, failures, errors, and the failing ids.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import io
import json
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg).resolve()))
    try:
        cand = importlib.import_module("mdifflib").SequenceMatcher
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"import_error": f"{type(e).__name__}: {e}", "all_pass": False}))
        return 1
    # Collect difflib's doctests before patching: once difflib.SequenceMatcher
    # is the candidate, DocTestFinder skips it (its __module__ is no longer
    # difflib) and the class's own doctests silently drop out (57 -> 48).
    # extraglobs makes those examples construct the candidate.
    import doctest
    doc_suite = doctest.DocTestSuite(difflib, extraglobs={"SequenceMatcher": cand})
    difflib.SequenceMatcher = cand

    # test_difflib imports test.support.findfile, which this interpreter lacks.
    support = types.ModuleType("test.support")
    support.findfile = lambda name, subdir=None: str(HERE / name)
    test_pkg = types.ModuleType("test")
    test_pkg.support = support
    sys.modules["test"] = test_pkg
    sys.modules["test.support"] = support
    sys.path.insert(0, str(HERE))
    import test_difflib  # noqa: E402

    test_difflib.load_tests = lambda loader, tests, pattern: tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_difflib)
    suite.addTest(doc_suite)
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    failing = [str(t) for t, _ in res.failures + res.errors]
    out = {
        "run": res.testsRun,
        "failures": len(res.failures),
        "errors": len(res.errors),
        "failing": failing[:20],
        "first_trace": (res.failures + res.errors)[0][1][-1500:] if failing else "",
        "all_pass": res.wasSuccessful(),
    }
    print(json.dumps(out))
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
