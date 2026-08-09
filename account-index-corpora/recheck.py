"""Check RESULTS.md against the artifacts it quotes.

The writeup and the data drift apart between rebuilds, and prose is where it
shows last. This reads the numbers back out of `corpora_scoped.json` and
`results.json` and asserts RESULTS.md still says them. Seconds, no deps.

    python3 account-index-corpora/recheck.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def main() -> None:
    prose = (HERE / "RESULTS.md").read_text()
    corp = json.loads((HERE / "corpora_scoped.json").read_text())
    depth = json.loads((HERE / "results.json").read_text())
    t = corp["totals"]

    print("corpus sizes")
    for key, label in (("tree", "tree"), ("tombstones", "tombstones"),
                       ("prs", "PR bodies")):
        n = t[key]["chunks"]
        check(f"{label} chunk count is quoted", f"{n:,}" in prose, f"{n:,}")
        pct = t[key]["pct_of_tree"]
        check(f"{label} share is quoted", f"{pct}%" in prose, f"{pct}%")

    print("per-repo inversion")
    per = corp["per_repo"]["claude-workspace"]
    check("claude-workspace tree count is quoted", str(per["tree"]) in prose)
    check("its tombstone count is quoted", f"{per['tombstones']:,}" in prose)
    check("tombstones really do exceed that tree", per["tombstones"] > per["tree"],
          f"{per['tombstones']} vs {per['tree']}")

    print("PR body profile")
    meds = sorted(v["median_body_chars"] for v in corp["pr_body_profile"].values())
    check("the quoted median range matches the data",
          f"{meds[0]:,}" in prose and f"{meds[-1]:,}" in prose,
          f"{meds[0]:,}-{meds[-1]:,}")
    empty = sum(v["under_min_chars"] for v in corp["pr_body_profile"].values())
    merged = sum(v["merged"] for v in corp["pr_body_profile"].values())
    check("the merged-PR count is quoted", str(merged) in prose, str(merged))
    check("the near-empty count is quoted", str(empty) in prose, str(empty))

    print("clone depth")
    by = {r["repo"].split("/")[1]: r for r in depth}
    tot = {k: round(sum(r[k]["secs"] for r in depth), 1)
           for k in ("depth1", "depth50", "full")}
    for k, v in tot.items():
        check(f"summed {k} seconds are quoted", f"{v} s" in prose, f"{v} s")
    m = by["muninn-utilities"]
    check("the muninn deletion-coverage claim matches",
          m["depth50"]["deleted_visible"] == 2 and m["deleted_full"] == 18,
          f"{m['depth50']['deleted_visible']}/{m['deleted_full']}")
    check("depth 50 really is not a saving",
          abs(tot["depth50"] - tot["full"]) < 0.5 * tot["depth1"],
          "the writeup says full clone is within noise of depth 50")

    print("account-wide run")
    acct = json.loads((HERE / "corpora_account.json").read_text())
    at = acct["totals"]
    for key, label in (("tree", "tree"), ("tombstones", "tombstones"),
                       ("prs", "PR bodies")):
        n, pct = at[key]["chunks"], at[key]["pct_of_tree"]
        check(f"{label} account chunk count is quoted", f"{n:,}" in prose, f"{n:,}")
        check(f"{label} account share is quoted", f"{pct}%" in prose, f"{pct}%")
    # the tree count is the load-bearing cross-check: `corpora` chunks through
    # the same path the build does, so it has to reproduce the published index
    check("account tree matches the published manifest's n_chunks",
          at["tree"]["chunks"] == 42578,
          "if this drifts, `corpora` is no longer measuring the real index")
    cov = acct["pr_coverage"]
    check("the PR floor is flagged rather than reported clean",
          "floor" in prose and f"{cov['repos_failed']} of 65" in prose,
          f"{cov['repos_failed']} repos failed")
    check("the 3-repo estimates are shown as revised, not quietly replaced",
          "11.8% →" in prose and "3.2% →" in prose)
    check("the account clone time is quoted",
          f"{acct['clone']['secs']} s" in prose, f"{acct['clone']['secs']} s")

    print("the corpus builder still applies the tree's exclusions")
    # the 74,822-chunk regression: without this the number in RESULTS.md is
    # history rather than a description of the code
    src = (HERE.parent / "hybrid-code-index" / "account.py").read_text()
    check("admissible() exists and is called by the tombstone path",
          "def admissible(" in src
          and re.search(r"admissible\(f, cfg\)", src) is not None)
    check("the size cap is checked against the retrieved body",
          'len(body.encode("utf-8", "ignore")) > cfg["max_bytes"]' in src)

    print()
    if FAILED:
        raise SystemExit(f"{len(FAILED)} failed: {', '.join(FAILED)}")
    print("all passed")


if __name__ == "__main__":
    main()
