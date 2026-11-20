"""Score a directory of page outputs with OmniParseBench's own scorer, on the text-family tests.

    python3 run/opb/score.py --run data/opb/out/tess_cli

RUN holds one <page_id>.md per subset page: the page's text in reading order, lines joined with
newlines, blocks separated by blank lines. Each .md goes through the benchmark's indra_ocr adapter
`parse` (markdown -> html, as the adapter does for every markdown output) and then `unit_tests.score`,
the calls opb_eval.py makes. Only the text family is scored (present, order, repeat, page_near_empty:
omni_parse_bench vocab.FAMILIES). A page with no .md, or whose parse fails, fails all its tests, as
opb_eval.py fails a page. The denominator is every text test of the subset.

Writes RUN/scored/pages/<page_id>.json (one test line per test) and RUN/scored/summary.json, and
prints the pass rate.

The benchmark package is code nobody here wrote, so the scoring runs INSIDE scripts/jail.sh: no
network, runs as nobody, an empty environment. The jail has no PYTHONPATH (`python3 -I` ignores it),
so the inner process puts the deps directory on sys.path itself. One-time setup:
run/opb/jail_setup.sh (as root).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "opb"
JAIL = Path("/home/user/claude-workspace/scripts/jail.sh")
JAIL_DIR = Path("/tmp/jail/opb-tg")
PREFIX = "@@ "


def inner(a: argparse.Namespace) -> None:
    """Runs inside the jail: the benchmark's adapter and scorer over each page. One JSON line per
    page and one for the summary, each after the PREFIX, so the outer process can find them."""
    sys.path.insert(0, a.deps)
    from types import SimpleNamespace

    from omni_parse_bench import data, vocab
    from omni_parse_bench.harness.providers import indra_ocr as P
    from omni_parse_bench.scorers import unit_tests
    from omni_parse_bench.scorers.unit_tests import tests as T

    subset = json.loads(Path(a.subset).read_text())
    run, ds = Path(a.run), Path(a.ds)
    every = []
    for row in subset["pages"]:
        pid = row["page_id"]
        page = data.Page(pid, row["document"], row["suite"], ds / row["doc_path"], ds / row["gt_path"],
                         ("html",))
        tests = tuple(t for t in T.read(page.gt_path) if vocab.FAMILIES[t.test_type] == "text")
        rec: dict = {"page_id": pid, "suite": row["suite"], "n_tests": len(tests)}
        try:
            md_path = run / f"{pid}.md"
            if not md_path.is_file():
                raise FileNotFoundError(f"no output file {pid}.md")
            md = md_path.read_text(encoding="utf-8")
            # The adapter reads markdown into html (text tests read html); a markdown-only page is fine.
            outputs, errs = P.parse({"markdown": md, "text": md}, SimpleNamespace(outputs=frozenset({"html"})))
            pred = {"html": outputs["html"]} if "html" in outputs else {}
            verdicts = unit_tests.score(pred, tests)
            rec["parse_errors"] = {k: str(v) for k, v in errs.items()}
        except Exception as e:  # noqa: BLE001 - a failed page fails all its tests (opb_eval.py)
            rec["error"] = f"{type(e).__name__}: {e}"
            verdicts = [unit_tests.Verdict(t.id, "fail", "page failed") for t in tests]
        rec["lines"] = unit_tests.lines(verdicts, tests, page)
        every += rec["lines"]
        print(PREFIX + json.dumps(rec, ensure_ascii=False), flush=True)
    summary = unit_tests.summarize(every)
    print(PREFIX + json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


def outer(a: argparse.Namespace) -> int:
    run = Path(a.run).resolve()
    if not run.is_dir():
        sys.exit(f"score.py: {run} is not a directory")
    subset_path = Path(a.subset).resolve()
    subset = json.loads(subset_path.read_text())
    cmd = ["sh", str(JAIL), "-C", str(a.jail_dir / "work"), "--",
           "python3", "-I", str(Path(__file__).resolve()), "--inner",
           "--deps", str(a.jail_dir / "deps"), "--run", str(run),
           "--subset", str(subset_path), "--ds", str(Path(a.ds).resolve())]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"score.py: the jailed scorer exited {proc.returncode}:\n{proc.stderr[-3000:]}")
    recs, summary = [], None
    for line in proc.stdout.splitlines():
        if not line.startswith(PREFIX):
            continue
        obj = json.loads(line[len(PREFIX):])
        if "summary" in obj:
            summary = obj["summary"]
        else:
            recs.append(obj)
    if summary is None or len(recs) != len(subset["pages"]):
        sys.exit(f"score.py: the jailed scorer returned {len(recs)} pages, not {len(subset['pages'])}")

    out = run / "scored"
    (out / "pages").mkdir(parents=True, exist_ok=True)
    for r in recs:
        (out / "pages" / f"{r['page_id']}.json").write_text(json.dumps(r, ensure_ascii=False))
    missing = sum(1 for r in recs if str(r.get("error", "")).startswith("FileNotFoundError"))
    failed = sum(1 for r in recs if "error" in r)
    tests = sum(r["n_tests"] for r in recs)
    passed = sum(ln["result"] == "pass" for r in recs for ln in r["lines"])
    text = summary["families"]["text"]
    if abs(text["score"] - passed / tests) > 1e-12:
        sys.exit("score.py: the summary's text score disagrees with the test lines")
    result = {"run": str(run), "subset": str(subset_path), "pages": len(recs), "pages_missing": missing,
              "failed_pages": failed, "tests": tests, "passed": passed, "pass_rate": passed / tests,
              "families": summary["families"], "per_suite": summary["per_suite"],
              "per_test_type": summary["per_test_type"]}
    (out / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"pass rate {passed / tests:.4f} ({passed}/{tests} text tests, {len(recs)} pages, "
          f"{missing} missing, {failed} failed) -> {out / 'summary.json'}")
    return 0


def main() -> int:
    if "--inner" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--inner", action="store_true")
        ap.add_argument("--deps", required=True)
        ap.add_argument("--run", required=True)
        ap.add_argument("--subset", required=True)
        ap.add_argument("--ds", required=True)
        inner(ap.parse_args())
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="a directory of <page_id>.md outputs")
    ap.add_argument("--subset", type=Path, default=DATA / "subset.json")
    ap.add_argument("--ds", type=Path, default=DATA / "ds")
    ap.add_argument("--jail-dir", type=Path, default=JAIL_DIR,
                    help="deps/ and work/ as run/opb/jail_setup.sh leaves them")
    return outer(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
