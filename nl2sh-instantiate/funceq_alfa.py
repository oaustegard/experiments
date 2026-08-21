#!/usr/bin/env python3
"""Execution scoring against the NL2SH benchmark's two-gold rows.

Three differences from `funceq_ext.py`, each because the benchmark is built
differently from the cyber corpus:

* **Two golds.** Every row carries `bash` and `bash2`, both acceptable. A
  prediction matching either is EQUIVALENT; a row is DIFFERENT only when it
  runs and matches neither.
* **`/testbed` is rewritten to the fixture root**, in the gold and the
  prediction alike, because the benchmark's own harness runs in a container
  where that path exists and this one does not. Rewriting both sides by the same
  rule keeps the comparison symmetric.
* **The fixture is built from both golds' paths**, so a command that reads a
  file the other gold writes still has it.

**This is not the paper's number.** The published 74% for GPT-4o comes from
InterCode-ALFA — their container image, their command set, and a model judge for
rows execution cannot separate. Here there is no judge: a row execution cannot
decide is INCONCLUSIVE and is excluded from the ratio rather than scored. So
what follows is an execution-only accuracy over the subset this sandbox can
decide, reported with its coverage, and it is a floor rather than a like-for-like
comparison.

    python3 funceq_alfa.py --results results_alfa_it_generate.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "nl2sh-retrieval"))

import funceq  # noqa: E402
from funceq_ext import path_tokens  # noqa: E402


def _cap_timeout(seconds: float) -> None:
    """Bound every subprocess in `funceq.run`.

    Its default is 10 s, and a benchmark row like `find / -name '*.log'` spends
    all of it. Two golds and a prediction per row over 270 rows makes that the
    difference between minutes and an hour, and a command that needs more than a
    few seconds in a fixture this small is not going to be decided by its output.
    """
    import functools
    if not getattr(funceq.run, "_capped", False):
        capped = functools.partial(funceq.run, timeout=seconds)
        capped._capped = True
        funceq.run = capped

TESTBED = "/testbed"


def build(rows: list[dict], root: Path) -> dict:
    funceq.build_fixture(root)
    made = {"files": 0, "dirs": 0}
    for r in rows:
        for gold in (r.get("gold_cmd"), r.get("alt_cmd")):
            if not gold:
                continue
            toks = gold.replace(TESTBED + "/", "").replace(TESTBED, "").split()
            for tok in path_tokens(" ".join(toks))[1:]:
                p = root / tok
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if not p.exists():
                        p.write_text(f"alpha beta\ncontents of {tok}\n")
                        made["files"] += 1
                except (OSError, ValueError):
                    continue
    return made


def rewrite(cmd: str, root: Path) -> str:
    return (cmd or "").replace(TESTBED, str(root))


def judge_row(r: dict, fixture: Path) -> dict:
    pred = rewrite(r.get("command", ""), fixture)
    verdicts = []
    for gold in (r.get("gold_cmd"), r.get("alt_cmd")):
        if not gold:
            continue
        v = funceq.judge(rewrite(gold, fixture), pred, fixture)
        if v["verdict"] == "EQUIVALENT":
            return {**v, "matched": gold}
        verdicts.append(v)
    for v in verdicts:
        if v["verdict"] == "DIFFERENT":
            return v
    return verdicts[0] if verdicts else {"verdict": "INCONCLUSIVE", "why": "no gold"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    _cap_timeout(a.timeout)
    report = {}
    for src in a.results:
        d = json.loads(src.read_text())
        rows = d["rows"] if a.all_rows else [r for r in d["rows"] if not r.get("names_utility")]
        fixture = Path(tempfile.mkdtemp(prefix="alfa-fx-"))
        made = build(rows, fixture)
        counts, whys, out = Counter(), Counter(), []
        for i, r in enumerate(rows, 1):
            v = judge_row(r, fixture)
            counts[v["verdict"]] += 1
            if v["verdict"] == "INCONCLUSIVE":
                whys[v.get("why", "")[:40]] += 1
            out.append({"i": i, "nl": r["nl"], "gold": r["gold_cmd"],
                        "alt": r.get("alt_cmd", ""), "pred": r.get("command", ""),
                        "utility_ok": r["utility_ok"], **v})
        shutil.rmtree(fixture, ignore_errors=True)
        decided = counts["EQUIVALENT"] + counts["DIFFERENT"]
        key = f"{Path(d['summary']['model']).name}/{d['summary']['condition']}"
        report[key] = {
            "n": len(rows), "fixture_added": made, **dict(counts),
            "coverage_decided": round(decided / len(rows), 3),
            "functional_acc_over_decided": round(counts["EQUIVALENT"] / decided, 3) if decided else None,
            "functional_acc_over_all": round(counts["EQUIVALENT"] / len(rows), 3),
            "routing_over_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3),
            "inconclusive_reasons": dict(whys.most_common(6)),
        }
        (src.parent / f"funceq_{src.stem}.json").write_text(
            json.dumps({"summary": report[key], "rows": out}, indent=1) + "\n")
        print(f"{key}: {json.dumps(report[key])}", flush=True)

    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
