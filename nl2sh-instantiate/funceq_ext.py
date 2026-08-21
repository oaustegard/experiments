#!/usr/bin/env python3
"""Run `nl2sh-retrieval/funceq.py` over a run, with a fixture built from the eval.

Stage 1 left functional equivalence blocked: 17 of 40 rows came back
INCONCLUSIVE because the fixture had none of the files the commands name. That
is the harness measuring its own sandbox, which `funceq.py` is careful to report
rather than score, and issue #52 makes unblocking it the first job.

The fix here is the first of the two routes the issue names: **widen the fixture
until the runnable subset is worth reporting**, and report coverage beside every
number. Every path a *gold* command names is created before anything runs —
files, their parent directories, and a directory for a path with a trailing
slash. Deriving the fixture from the gold side only keeps it neutral: both
commands meet the same tree, and a prediction that invents a different filename
still fails, which is the behaviour wanted.

Absolute paths are not created (`/etc/services` stays absent, and its command
stays INCONCLUSIVE), interactive editors time out into INCONCLUSIVE, and the
deny list is untouched. So this widens coverage without widening what counts as
a pass.

    python3 funceq_ext.py --results results_it_instantiate.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nl2sh-retrieval"))

import funceq  # noqa: E402


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

# A token that looks like a path or a filename: has a slash or a dot-extension,
# or is a bare word in argument position. Flags, globs, and options are skipped.
PATHY = re.compile(r"^[A-Za-z0-9._~\-][A-Za-z0-9._/\-+~]*$")


def path_tokens(cmd: str) -> list[str]:
    out = []
    for tok in cmd.split():
        tok = tok.strip("'\"`;")
        if not tok or tok.startswith("-") or tok.startswith("/") or "*" in tok or "$" in tok:
            continue
        if tok in ("~", ".", "..") or "=" in tok:
            continue
        if PATHY.match(tok) and ("/" in tok or "." in tok or tok.isidentifier()
                                 or re.match(r"^[A-Za-z0-9_.\-]+$", tok)):
            out.append(tok)
    return out


def build(rows: list[dict], root: Path) -> dict:
    """funceq's fixture, plus every relative path the gold commands name."""
    funceq.build_fixture(root)
    made = {"files": 0, "dirs": 0}
    for r in rows:
        gold = r.get("gold_cmd") or ""
        # argv[0] is the utility, never a file to create
        for tok in path_tokens(gold)[1:] if gold.split() else []:
            p = root / tok
            try:
                if tok.endswith("/"):
                    p.mkdir(parents=True, exist_ok=True)
                    made["dirs"] += 1
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text(f"alpha beta\ncontents of {tok}\n")
                    made["files"] += 1
            except (OSError, ValueError):
                continue
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    _cap_timeout(a.timeout)
    report = {}
    for src in a.results:
        d = json.loads(src.read_text())
        rows = d["rows"] if a.all_rows else [r for r in d["rows"] if not r.get("names_utility")]
        fixture = Path(tempfile.mkdtemp(prefix="funceq-fx-"))
        made = build(rows, fixture)
        counts, whys, out = Counter(), Counter(), []
        for i, r in enumerate(rows, 1):
            v = funceq.judge(r["gold_cmd"], r.get("command", ""), fixture)
            counts[v["verdict"]] += 1
            if v["verdict"] == "INCONCLUSIVE":
                whys[v.get("why", "")[:40]] += 1
            out.append({"i": i, "nl": r["nl"], "gold": r["gold_cmd"],
                        "pred": r.get("command", ""), "utility_ok": r["utility_ok"], **v})
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
        dest = src.parent / f"funceq_{src.stem}.json"
        dest.write_text(json.dumps({"summary": report[key], "rows": out}, indent=1) + "\n")
        print(f"{key}: {json.dumps(report[key])}", flush=True)

    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
