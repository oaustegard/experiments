"""Certify the grader on each library in both directions before any agent runs.

    python harness/certify.py [NAME ...]

For each library: the stub tree (base commit) and the reference tree (reference
src_dir overlaid on the base commit, so both see the same tests) run the suite in
the jail. A library is admissible when the reference passes a majority of its
tests and the stub passes fewer than the reference. The tests the reference passes
are the library's target set; those the stub fails are its headroom set; data/targets/NAME.json, summary in data/certify.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, WORK, git, load_tasks  # noqa: E402
from workspace import checkout, overlay_src, run_tests  # noqa: E402

CERT = WORK / "cert"


def certify(task: dict) -> dict:
    n = task["name"]
    stub = checkout(task, CERT / n / "stub")
    ref_src = checkout(task, CERT / n / "ref-src", commit=task["reference_commit"])
    ref = checkout(task, CERT / n / "ref")
    overlay_src(task, ref_src, ref)
    nonsrc = git("diff", "--name-only", task["base_commit"], task["reference_commit"], cwd=stub).split()
    nonsrc = [f for f in nonsrc if not f.startswith(task["src_dir"])]
    rs, ss = run_tests(task, ref), run_tests(task, stub)
    out = {"name": n, "nonsrc_diff": nonsrc, "ref_exit": rs["exit"], "stub_exit": ss["exit"]}
    if rs["outcomes"] is None:
        out.update(admissible=False, reason="reference produced no junit", ref_tail=rs["tail"][-1500:])
        return out
    ro, so = rs["outcomes"], ss["outcomes"] or {}
    target = sorted(k for k, v in ro.items() if v == "pass")
    stub_pass = sum(so.get(k) == "pass" for k in target)
    out.update(n_tests=len(ro), ref_pass=len(target), ref_fail=sum(v == "fail" for v in ro.values()),
               stub_pass_of_target=stub_pass, stub_collected=len(so))
    frac_ref = len(target) / max(1, len(ro))
    out["admissible"] = bool(target) and frac_ref >= 0.5 and stub_pass < len(target)
    if not out["admissible"]:
        out["reason"] = f"ref {len(target)}/{len(ro)}, stub {stub_pass}/{len(target)}"
        out["ref_tail"] = rs["tail"][-1500:]
    (DATA / "targets").mkdir(exist_ok=True)
    (DATA / "targets" / f"{n}.json").write_text(json.dumps(target, indent=0) + "\n")
    stub_passing = sorted(k for k in target if so.get(k) == "pass")
    (DATA / "targets" / f"{n}.stubpass.json").write_text(json.dumps(stub_passing, indent=0) + "\n")
    return out


def main(names: list[str]):
    tasks = load_tasks()
    names = names or sorted(tasks)
    path = DATA / "certify.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda n: certify(tasks[n]), names):
            summary[r["name"]] = r
            brief = {k: r.get(k) for k in ("admissible", "n_tests", "ref_pass", "stub_pass_of_target", "reason")}
            print(r["name"], json.dumps(brief), flush=True)
            path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
