"""Check every number RESULTS.md quotes against results/summary.json (seconds, no network).

    python3 recheck.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
s = json.load(open(os.path.join(HERE, "results", "summary.json")))
doc = open(os.path.join(HERE, "RESULTS.md")).read()
a, f = s["arms"], s["filter"]
checks = [
    (f"{a['jev1']['facts_found']}/{s['facts']} (94%)", "jev1 facts"),
    (f"{a['full']['facts_found']}/{s['facts']} (97%)", "full facts"),
    (f"{a['brief']['facts_found']}/{s['facts']} (78%)", "brief facts"),
    (f"| jev1 | {a['jev1']['facts_found']}/{s['facts']} (94%) | {a['jev1']['tasks_fully_correct']}/{s['tasks']}", "jev1 row"),
    (f"| jev4 | {a['jev4']['facts_found']}/{s['facts']} (94%) | {a['jev4']['tasks_fully_correct']}/{s['tasks']}", "jev4 row"),
    (f"| full | {a['full']['facts_found']}/{s['facts']} (97%) | {a['full']['tasks_fully_correct']}/{s['tasks']}", "full row"),
    (f"| brief | {a['brief']['facts_found']}/{s['facts']} (78%) | {a['brief']['tasks_fully_correct']}/{s['tasks']}", "brief row"),
    (f"jev1 {a['jev1']['facts_found_regex_only']}, jev4 {a['jev4']['facts_found_regex_only']}, "
     f"full {a['full']['facts_found_regex_only']}, brief {a['brief']['facts_found_regex_only']}", "regex-only"),
    (f"{f['jev1']['must_kept']}/{f['jev1']['must_total']}", "jev1 must-have"),
    (f"{f['jev4']['must_kept']}/{f['jev4']['must_total']}", "jev4 must-have"),
    (f"{s['adjudicator']['upgrades']} of {s['adjudicator']['misses']} misses", "adjudicator upgrades"),
    (f"TPR {s['adjudicator']['tpr']:.0%}", "TPR"), (f"TNR {s['adjudicator']['tnr']:.0%}", "TNR"),
    (f"median {int(s['brief_writer']['output_tokens_median'])} output tokens", "brief tokens"),
    (f"{s['tasks']} tasks, {s['facts']} facts", "counts"),
]
bad = [name for needle, name in checks if needle not in doc]
# the 27-task exclusion line is recomputed from per-task counts
excl = ["c8294492-3", "d113561d-4", "5bde6e59-3", "8e52c171-3", "5bde6e59-4"]
sub = {arm: (sum(v[arm][0] for t, v in s["per_task"].items() if t not in excl),
             sum(v[arm][1] for t, v in s["per_task"].items() if t not in excl)) for arm in ("jev1", "jev4", "full", "brief")}
line = ", ".join(f"{arm} {x}/{n}" for arm, (x, n) in sub.items())
if line not in doc:
    bad.append(f"exclusion line (expected '{line}')")
print("recheck:", "OK" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
