"""Score a chunked grade still in progress from the chunk junits it has written so far.

    python harness/partial.py RUN [LIB]     (LIB defaults to statsmodels)

A lower bound on the final score: tests in unfinished chunks count as not passed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, WORK  # noqa: E402
from run import normalize_outcomes, param_key  # noqa: E402
from workspace import parse_junit  # noqa: E402


def main(run: str, lib: str = "statsmodels"):
    g = WORK / f"grade-{run}" / lib
    oc = {}
    files = sorted(g.glob(".c0-junit.xml.*"))
    for f in files:
        oc.update(parse_junit(f) or {})
    oc = normalize_outcomes(oc)
    target = json.loads((DATA / "targets" / f"{lib}.json").read_text())
    passed = sum(oc.get(param_key(k)) == "pass" for k in target)
    print(f"{run} {lib}: {passed} / {len(target)} passed in {len(files)} finished chunks "
          f"(lower bound; {len(oc)} outcomes seen)")


if __name__ == "__main__":
    main(*sys.argv[1:])
