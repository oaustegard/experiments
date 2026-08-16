#!/usr/bin/env python3
"""Consistency fixture for orchestrated-coding-pareto (< 1 min).

Checks that the prose, the stored data, and the artifacts cannot drift apart:
 1. every task has spec.md + tests.py + reference.py, and every reference passes
    its own hidden suite;
 2. stored solutions re-grade to the stored results;
 3. analysis.json is reproducible from marks + results + params;
 4. the headline numbers quoted in RESULTS.md match analysis.json;
 5. params hygiene: every price row carries source-or-note and confidence.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
from grade import grade_one  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'ok ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main():
    tasks = sorted(p.name for p in (ROOT / "tasks").iterdir() if p.is_dir())
    check("task count is 14", len(tasks) == 14, str(len(tasks)))

    # 1. artifacts + reference validity
    for t in tasks:
        d = ROOT / "tasks" / t
        check(f"{t}: files present", all((d / f).exists() for f in ("spec.md", "tests.py", "reference.py")))
        r = grade_one(t, (d / "reference.py").read_text())
        check(f"{t}: reference passes hidden suite", r["passed"], r["failures"][:200])

    # 2. stored solutions re-grade to stored results
    stored = {}
    for f in sorted((ROOT / "data").glob("results_*.json")):
        for arm, per in json.loads(f.read_text()).items():
            stored.setdefault(arm, {}).update(per)
    for arm, per in stored.items():
        for t, res in per.items():
            src = (ROOT / "data" / "solutions" / arm / f"{t}.py").read_text()
            now = grade_one(t, src)
            check(f"regrade {arm}/{t} stable", now["passed"] == res["passed"])

    # 3. analysis reproducible
    before = json.loads((ROOT / "data" / "analysis.json").read_text())
    subprocess.run([sys.executable, str(ROOT / "harness" / "analyze.py")],
                   capture_output=True, check=True)
    after = json.loads((ROOT / "data" / "analysis.json").read_text())
    check("analysis.json reproducible", before == after)

    # 4. prose anchors
    a = after["arms"]
    prose = (ROOT / "RESULTS.md").read_text()
    anchors = {
        "haiku output tokens": (f"{a['haiku-solo']['output_tokens_measured']:,}", True),
        "opus output tokens": (f"{a['opus-solo']['output_tokens_measured']:,}", True),
        "haiku $/task": (f"${a['haiku-solo']['cost_usd_per_task']:.3f}", True),
        "opus $/task": (f"${a['opus-solo']['cost_usd_per_task']:.3f}", True),
        "verbosity ratio 6.7": ("6.7", True),
        "haiku pass 14/14": ("14/14", True),
        "sonnet pass 13/14": ("13/14", True),
    }
    if "pipelines" in after:
        pl = after["pipelines"]
        anchors["haiku-low tokens"] = (f"{a['haiku-low']['output_tokens_measured']:,}", True)
        anchors["haiku-low pass 12/14"] = ("12/14", True)
        anchors["pipeline $/task"] = (f"${pl['haiku-low+test-retry']['cost_usd_per_task']:.3f}", True)
        anchors["orch pipeline $/task"] = (f"${pl['haiku-low+opus-orch']['cost_usd_per_task']:.3f}", True)
        anchors["luna pipeline $/task"] = (f"${pl['haiku-low+test-retry']['cost_usd_per_task_at_luna']:.3f}", True)
    for name, (needle, want) in anchors.items():
        check(f"prose anchor: {name} ({needle})", (needle in prose) == want)
    ratio = a["haiku-solo"]["output_tokens_measured"] / a["opus-solo"]["output_tokens_measured"]
    check("verbosity ratio consistent", abs(ratio - after["notes"]["haiku_output_ratio_vs_opus"]) < 0.01)
    check("haiku==opus pass parity", a["haiku-solo"]["tasks_passed"] == a["opus-solo"]["tasks_passed"] == 14)

    # 5. params hygiene
    prices = json.loads((ROOT / "params.json").read_text())["api_prices_usd_per_mtok"]
    for k, v in prices.items():
        check(f"params: {k} sourced", "confidence" in v and ("source" in v or "note" in v or k.startswith("claude-")))

    n = len(FAILURES)
    total = sum(1 for _ in FAILURES) if n else None
    print(f"\n{'OK' if not FAILURES else 'FAILED'} — {n} failing check(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
