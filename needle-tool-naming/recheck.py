#!/usr/bin/env python3
"""Check every number quoted in RESULTS.md against the result artifacts.

    python3 recheck.py        # exits non-zero on the first disagreement

Sub-5-minute fixture, no network and no model: it re-derives the claims from
`results_*.json` rather than trusting the prose. The point is that the writeup
and the data cannot drift apart between full rebuilds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import load, mcnemar, per_category, routable  # noqa: E402

FAILURES: list[str] = []


def eq(label: str, got, want, tol: float = 5e-4) -> None:
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        FAILURES.append(f"{label}: RESULTS.md says {want}, artifacts say {got}")
    print(f"  {'ok ' if ok else 'FAIL'} {label:58} {got}")


def main() -> int:
    print("routable top-1, flat 18")
    for v, want in [("canon", 0.611), ("desc-only", 0.407), ("names-only", 0.444),
                    ("neither", 0.074), ("separated", 0.611), ("adversarial", 0.426)]:
        eq(f"{v} flat", load(v, "flat")["summary"]["tool_acc_routable"], want)

    print("routable top-1, oracle 5")
    for v, want in [("canon", 0.778), ("desc-only", 0.463), ("names-only", 0.630),
                    ("neither", 0.259), ("separated", 0.778), ("adversarial", 0.704)]:
        eq(f"{v} oracle", load(v, "oracle")["summary"]["tool_acc_routable"], want)

    print("retrieval cost (oracle - flat)")
    for v, want in [("canon", 0.167), ("desc-only", 0.056), ("names-only", 0.185),
                    ("adversarial", 0.278)]:
        cost = (load(v, "oracle")["summary"]["tool_acc_routable"]
                - load(v, "flat")["summary"]["tool_acc_routable"])
        eq(f"{v} retrieval cost", round(cost, 3), want, tol=1e-3)

    print("channel decomposition against the `neither` floor")
    floor = load("neither", "flat")["summary"]["tool_acc_routable"]
    names = load("names-only", "flat")["summary"]["tool_acc_routable"] - floor
    desc = load("desc-only", "flat")["summary"]["tool_acc_routable"] - floor
    both = load("canon", "flat")["summary"]["tool_acc_routable"] - floor
    eq("names alone", round(names, 3), 0.370, tol=1e-3)
    eq("descriptions alone", round(desc, 3), 0.333, tol=1e-3)
    eq("both together", round(both, 3), 0.537, tol=1e-3)
    eq("sum of the two channels", round(names + desc, 3), 0.704, tol=1e-3)

    print("paired McNemar")
    for a, b, mode, wa, wb, wp in [
        ("names-only", "desc-only", "flat", 11, 9, 0.82),
        ("names-only", "desc-only", "oracle", 16, 7, 0.093),
        ("canon", "names-only", "flat", 12, 3, 0.035),
        ("canon", "desc-only", "flat", 15, 4, 0.019),
        ("canon", "adversarial", "flat", 19, 9, 0.087),
        ("canon", "separated", "flat", 5, 5, 1.00),
        ("canon", "separated", "oracle", 3, 3, 1.00),
    ]:
        ao, bo, p = mcnemar(routable(load(a, mode)), routable(load(b, mode)))
        eq(f"{a} vs {b} ({mode}) discordant", (ao, bo), (wa, wb))
        eq(f"{a} vs {b} ({mode}) p", round(p, 3), wp, tol=6e-3)
    ao, bo, p = mcnemar(routable(load("canon", "flat")), routable(load("neither", "flat")))
    eq("canon vs neither discordant", (ao, bo), (30, 1))
    eq("canon vs neither p < 0.0001", p < 1e-4, True)

    print("the profile category and its two flipped queries")
    eq("profile canon flat", per_category(load("canon", "flat"))["profile"], 0.250)
    eq("profile adversarial flat", per_category(load("adversarial", "flat"))["profile"], 0.750)
    rows = {r["query"]: r for r in load("adversarial", "flat")["rows"]}
    canon_rows = {r["query"]: r for r in load("canon", "flat")["rows"]}
    for q, canon_tool, canon_c, adv_c in [
        ("how many followers does pfrazee.com have", "get_followers", 0.80, 0.81),
        ("look up the account jay.bsky.team", "get_user_posts", 0.75, 0.90),
    ]:
        eq(f"canon tool for {q!r}", canon_rows[q]["got"], canon_tool)
        eq(f"canon confidence for {q!r}", round(canon_rows[q]["confidence"], 2), canon_c, tol=6e-3)
        eq(f"adversarial tool for {q!r}", rows[q]["got"], "get_profile")
        eq(f"adversarial confidence for {q!r}", round(rows[q]["confidence"], 2), adv_c, tol=6e-3)
    eq("identity name/description conflict keeps the answer",
       rows["resolve did:plc:s3cqfxbcwnvvyrsttl3wivgp to a handle"]["got"], "resolve_identity")
    eq("...and loses the confidence",
       round(rows["resolve did:plc:s3cqfxbcwnvvyrsttl3wivgp to a handle"]["confidence"], 3), 0.167)
    eq("canon confidence on the same query",
       round(canon_rows["resolve did:plc:s3cqfxbcwnvvyrsttl3wivgp to a handle"]["confidence"], 3), 0.584,
       tol=6e-4)

    print("gate separation and latency")
    for v, want in [("canon", 0.191), ("separated", 0.101)]:
        s = load(v, "flat")["summary"]
        eq(f"{v} conf separation", round(s["mean_conf_correct"] - s["mean_conf_wrong"], 3), want, tol=1e-3)
    eq("separated mean confidence on wrong calls",
       round(load("separated", "flat")["summary"]["mean_conf_wrong"], 3), 0.480, tol=1e-3)
    eq("canon median turn", load("canon", "flat")["summary"]["median_latency_ms"], 808.0, tol=1.0)
    eq("names-only median turn", load("names-only", "flat")["summary"]["median_latency_ms"], 532.0, tol=1.0)

    print("cross-checks against needle-bsky")
    from names import NEEDLE_BSKY
    nb = lambda arm: json.loads((NEEDLE_BSKY / f"results_{arm}.json").read_text())
    eq("needle-bsky tuned-min routable == canon", nb("tuned-min")["summary"]["tool_acc_routable"],
       load("canon", "flat")["summary"]["tool_acc_routable"])
    eq("needle-bsky oracle auto -> tuned gain",
       round(nb("oracle-tuned")["summary"]["tool_acc_routable"]
             - nb("oracle-auto")["summary"]["tool_acc_routable"], 3), 0.204, tol=1e-3)
    eq("needle-bsky flat auto -> tuned gain",
       round(nb("tuned")["summary"]["tool_acc_routable"]
             - nb("auto")["summary"]["tool_acc_routable"], 3), 0.259, tol=1e-3)
    eq("needle-bsky oracle-tuned ceiling", nb("oracle-tuned")["summary"]["tool_acc_routable"], 0.815)
    eq("needle-bsky flat tuned", nb("tuned")["summary"]["tool_acc_routable"], 0.704)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} disagreement(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULTS.md agrees with the artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
