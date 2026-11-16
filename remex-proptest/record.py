#!/usr/bin/env python3
"""Rerun each area's regression file against remex 2.2.0 and the fixed commit,
and tally the planted-bug (mutant) results each area recorded.

    python3 record.py /path/to/remex-clone    # needs both commits fetched

Writes results/summary.json. Regression runs take about a minute; the
property harnesses are not rerun here (each takes 2-5 minutes; see Method).
"""
import json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD, NEW = "b8d527a", "0319e2d"
AREAS = {"search": "test_search_regressions.py",
         "containers": "test_containers_regressions.py",
         "stateful": "test_stateful_regressions.py",
         "encode": "test_encode_regressions.py"}

def run(area, src):
    env = {**os.environ, "PYTHONPATH": src}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-W", "ignore", AREAS[area]], cwd=HERE / area, env=env,
                       capture_output=True, text=True, timeout=1200)
    tail = r.stdout.strip().splitlines()[-1]
    return {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|xfailed|xpassed)", tail)}

def mutants():
    out = {}
    t = (HERE / "search/MUTATION_RESULTS.md").read_text()
    m = re.search(r"(\d+)/(\d+) mutants caught", t); out["search"] = [int(m[1]), int(m[2])]
    t = (HERE / "containers/mutation_results.txt").read_text()
    rows = re.findall(r"^(M\d+)\s+(CAUGHT|SURVIVED)", t, re.M)
    first = {}
    for mid, res in rows: first.setdefault(mid, res)   # first table = full harness run
    out["containers"] = [sum(v == "CAUGHT" for v in first.values()), len(first)]
    j = json.loads((HERE / "stateful/results/mutation_results.json").read_text())
    ms = j["mutants"] if isinstance(j, dict) and "mutants" in j else j
    caught = [bool(x.get("caught")) for x in (ms.values() if isinstance(ms, dict) else ms)
              if x.get("id") != "baseline"]  # baseline rows are the unmutated control
    out["stateful"] = [sum(caught), len(caught)]
    t = (HERE / "encode/mutation_results.md").read_text()
    m = re.search(r"All (\d+) caught", t); out["encode"] = [int(m[1]), int(m[1])]
    return out

def main():
    repo = Path(sys.argv[1]).resolve()
    res = {"old_commit": OLD, "new_commit": NEW, "regressions": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for tag, sha in (("old", OLD), ("new", NEW)):
            wt = Path(tmp) / tag
            subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(wt), sha], check=True)
            try:
                for area in AREAS:
                    res["regressions"].setdefault(area, {})[tag] = run(area, str(wt))
            finally:
                subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)])
    res["mutants_caught_of"] = mutants()
    res["totals"] = {
        "regression_failures_on_old": sum(r["old"].get("failed", 0) for r in res["regressions"].values()),
        "regression_failures_on_new": sum(r["new"].get("failed", 0) for r in res["regressions"].values()),
        "mutants_caught": sum(v[0] for v in res["mutants_caught_of"].values()),
        "mutants_planted": sum(v[1] for v in res["mutants_caught_of"].values()),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/summary.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
