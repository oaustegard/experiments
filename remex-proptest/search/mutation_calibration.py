"""Run every planted mutant (mutants.py) against the harness and record which
properties caught it.  A mutant is "caught" when any property that passes on
pristine remex fails under it.  The three properties that fail on pristine
remex (they pin real findings) are deselected so they cannot mask a survivor.

    python mutation_calibration.py [--examples 300] [--jobs 2] [--only name,...]

Writes MUTATION_RESULTS.md next to this file.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants  # noqa: E402

BASELINE_FAILING = ["test_mismatched_reader_is_refused_on_every_path", "test_ivf_flat_limit_is_byte_identical_to_adc",
                    "test_score_vectors_accepts_packed",
                    "test_query_offset_cannot_reorder"]
DESELECT = " and ".join(f"not {t}" for t in BASELINE_FAILING)


def run(name, examples):
    env = dict(os.environ, REMEX_PT_MUTANT=name, REMEX_PT_EXAMPLES=str(examples),
               PYTHONDONTWRITEBYTECODE="1")
    t = time.time()
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_search_props.py",
         "-k", DESELECT, "--tb=line"],
        cwd=HERE, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    failed = sorted(set(re.findall(r"^FAILED test_search_props.py::(\w+)", out, re.M)))
    first = ""
    m = re.search(r"Violation: (.{0,220})", out)
    if m:
        first = m.group(1).strip()
    else:
        m = re.search(r"^E\s+(\w*(?:Error|Exception)[^\n]{0,200})", out, re.M)
        if m:
            first = m.group(1)
    return name, failed, first, time.time() - t, out[-400:] if p.returncode not in (0, 1) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in mutants.MUTANTS if not a.only or n in a.only.split(",")]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda n: run(n, a.examples), names))
    lines = ["# Mutation calibration, area A (search-path agreement)", "",
             f"Harness `test_search_props.py`, {a.examples} examples per property, pristine-failing "
             f"properties deselected ({', '.join(BASELINE_FAILING)}).", "",
             "| mutant | planted defect | caught | by |", "|---|---|---|---|"]
    caught = 0
    for name, failed, first, secs, tail in results:
        ok = bool(failed)
        caught += ok
        lines.append(f"| `{name}` | {mutants.MUTANTS[name][0]} | {'yes' if ok else '**NO (survived)**'} | "
                     f"{', '.join(failed) if failed else '-'} |")
    lines += ["", f"**{caught}/{len(results)} mutants caught.**", "", "## First violation per mutant", ""]
    for name, failed, first, secs, tail in results:
        lines.append(f"- `{name}` ({secs:.0f}s): {first or '(none)'}" + (f"  [runner tail: {tail!r}]" if tail else ""))
    (HERE / "MUTATION_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
