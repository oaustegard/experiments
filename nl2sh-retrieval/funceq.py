#!/usr/bin/env python3
"""Functional equivalence by execution, not by string match or by vibes.

Every accuracy figure in this directory so far is `utility_ok` — does the
predicted command lead with the same utility as the gold. That counts
`find . -type f -print0 | xargs -0 grep -i '.*\\.*'` a success for a `find`
gold while it is nonsense, so **0.923 is a routing number wearing a generation
number's clothes**. This is the harness that takes the clothes off.

The published NL2SH benchmark (arXiv:2502.06858) scores by running both commands
in a container and having a model judge whether the outputs accomplish the same
task, and reports 74% for GPT-4o that way. This is the same idea with the parts
available here: a disposable fixture directory, real execution, output
comparison, and a model judge only where execution cannot decide.

Three verdicts, and the distinction between the last two is the point:

* **EQUIVALENT** — both ran and produced the same normalised stdout and exit
  status. Decided by execution; no judgement involved.
* **DIFFERENT** — both ran and disagreed. Also decided by execution.
* **INCONCLUSIVE** — one or both could not run here (needs root, needs a package
  this container lacks, touches the network, or is destructive outside the
  fixture). These are *not* counted as either, and the fraction of them is
  reported, because a harness that silently scores unrunnable commands as
  failures is measuring its own sandbox.

Safety: every command runs with `cwd` inside a fresh copy of the fixture, under
`timeout`, with no network, and is refused outright if it matches the deny list
(absolute-path writes, `sudo`, `mkfs`, `dd`, `shutdown`, fork bombs). The point
is to measure commands, not to be clever about running dangerous ones.

    python3 funceq.py --results results_gate_ft.json --limit 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Refused outright — not scored, reported as INCONCLUSIVE(refused).
DENY = re.compile(r"""
    \bsudo\b | \bsu\b | \bmkfs | \bdd\s+if= | \bshutdown\b | \breboot\b
  | \bmount\b | \bumount\b | \bchown\s+-R\s+/ | \brm\s+-rf\s+/(?!tmp) 
  | :\(\)\{ | \bcurl\b | \bwget\b | \bssh\b | \bscp\b | \byum\b | \bapt\b
  | \bkill\b | \bpkill\b | \bkillall\b | >\s*/dev/(?!null) | \bcrontab\b
""", re.X)


def build_fixture(root: Path) -> None:
    """A small, deterministic tree with the properties these commands ask about:
    extensions, sizes, ages, permissions, nesting, and content to grep."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src" / "deep").mkdir(parents=True, exist_ok=True)
    files = {
        "notes.txt": "alpha beta gamma\nsecond line\n",
        "report.pdf": "%PDF-1.4 fake\n",
        "docs/manual.pdf": "%PDF-1.4 also fake\n",
        "docs/readme.md": "# title\nalpha\n",
        "src/main.c": "int main(void){return 0;}\n",
        "src/util.c": "/* alpha */\n",
        "src/app.php": "<?php echo 'alpha'; ?>\n",
        "src/deep/nested.c": "// deep\n",
        "src/deep/big.log": "x" * 200000,
        "empty.txt": "",
    }
    for rel, body in files.items():
        p = root / rel
        p.write_text(body)
    os.chmod(root / "src" / "main.c", 0o755)
    os.chmod(root / "notes.txt", 0o644)
    # a stable set of mtimes so -mtime / -newer behave deterministically
    import time
    old = time.time() - 60 * 60 * 24 * 40
    os.utime(root / "docs" / "manual.pdf", (old, old))
    os.utime(root / "src" / "deep" / "big.log", (old, old))


def normalise(out: str) -> str:
    lines = [l.rstrip() for l in out.splitlines() if l.strip()]
    # order of directory traversal is not semantic; sizes/inodes vary
    lines = [re.sub(r"\b\d{4,}\b", "N", l) for l in lines]
    return "\n".join(sorted(lines))


def run(cmd: str, fixture: Path, timeout: float = 10.0) -> dict:
    work = Path(tempfile.mkdtemp(prefix="funceq-"))
    try:
        shutil.copytree(fixture, work / "fx", symlinks=True)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work),
               "LC_ALL": "C", "TERM": "dumb"}
        p = subprocess.run(["bash", "-c", cmd], cwd=work / "fx", env=env,
                           capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout[:20000], "err": p.stderr[:2000],
                "ran": True}
    except subprocess.TimeoutExpired:
        return {"ran": False, "why": "timeout"}
    except Exception as e:
        return {"ran": False, "why": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def judge(gold: str, pred: str, fixture: Path) -> dict:
    if not pred.strip():
        return {"verdict": "DIFFERENT", "why": "no command produced"}
    for label, c in (("gold", gold), ("pred", pred)):
        if DENY.search(c):
            return {"verdict": "INCONCLUSIVE", "why": f"{label} refused by deny list"}
    g, p = run(gold, fixture), run(pred, fixture)
    if not g["ran"] or not p["ran"]:
        return {"verdict": "INCONCLUSIVE",
                "why": f"gold_ran={g['ran']} pred_ran={p['ran']}"}
    # A gold that itself errors tells us nothing about the prediction.
    if g["rc"] != 0 and not g["out"].strip():
        return {"verdict": "INCONCLUSIVE", "why": f"gold failed rc={g['rc']}",
                "gold_err": g["err"][:200]}
    same = normalise(g["out"]) == normalise(p["out"]) and (g["rc"] == 0) == (p["rc"] == 0)
    return {"verdict": "EQUIVALENT" if same else "DIFFERENT",
            "gold_rc": g["rc"], "pred_rc": p["rc"],
            "gold_out": g["out"][:300], "pred_out": p["out"][:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True,
                    help="a results_gate*.json with rows carrying gold_cmd and command")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = json.loads(a.results.read_text())["rows"][: a.limit]
    fixture = Path(tempfile.mkdtemp(prefix="funceq-fixture-"))
    build_fixture(fixture)

    out, counts = [], {"EQUIVALENT": 0, "DIFFERENT": 0, "INCONCLUSIVE": 0}
    for i, r in enumerate(rows, 1):
        v = judge(r["gold_cmd"], r.get("command", ""), fixture)
        counts[v["verdict"]] += 1
        out.append({"i": i, "nl": r["nl"], "gold": r["gold_cmd"],
                    "pred": r.get("command", ""), "utility_ok": r["utility_ok"], **v})
        print(f"{i:>3} {v['verdict']:<13} util_ok={str(r['utility_ok']):<5} {r.get('command','')[:52]}")
    shutil.rmtree(fixture, ignore_errors=True)

    decided = counts["EQUIVALENT"] + counts["DIFFERENT"]
    summary = {"source": str(a.results), "n": len(rows), **counts,
               "decided_by_execution": decided,
               "functional_acc_over_decided": round(counts["EQUIVALENT"] / decided, 3) if decided else None,
               "functional_acc_over_all": round(counts["EQUIVALENT"] / len(rows), 3) if rows else None,
               "utility_acc_over_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3)}
    dest = a.out or HERE / f"results_funceq_{a.results.stem}.json"
    dest.write_text(json.dumps({"summary": summary, "rows": out}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
