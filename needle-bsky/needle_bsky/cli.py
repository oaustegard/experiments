#!/usr/bin/env python3
"""Natural-language front end to the Bluesky read tools, routed by Needle 2.

    python3 -m needle_bsky route "who liked <post url>"     # decide only, no network
    python3 -m needle_bsky ask   "what has pfrazee.com been posting"
    python3 -m needle_bsky repl                             # keep the agent warm
    python3 -m needle_bsky ask "..." --router flat          # declare all 18 instead

`route` is the measured layer: one 45M-parameter turn, no Bluesky credentials
needed. `ask` adds the confidence gate and executes the chosen tool.

The default router is the two-stage one: a deterministic group pick, then a
≤5-tool Needle agent. On the eval set that is 0.722 routable at 316 ms against
0.611 at 1187 ms for declaring all 18 to one agent (`--router flat`).

Below the threshold nothing runs. The exit code says so — 3 means "escalate",
i.e. hand this query to a larger model rather than guessing at a network call.
"""

from __future__ import annotations

import argparse
import json

from .router import ARMS, DEFAULT_THRESHOLD, Router

EXIT_OK, EXIT_ERR, EXIT_REFUSED, EXIT_ESCALATE = 0, 1, 2, 3


def _fmt_decision(d) -> str:
    conf = "n/a" if d.confidence is None else f"{d.confidence:.3f}"
    if d.tool is None:
        return f"[refuse] no declared tool serves this (confidence {conf}, {d.latency_ms:.0f} ms)"
    args = ", ".join(f"{k}={v!r}" for k, v in d.arguments.items())
    return f"[{d.decision}] {d.tool}({args})  confidence {conf}  {d.latency_ms:.0f} ms"


def _digest(result, limit: int = 8) -> str:
    if result is None:
        return ""
    if isinstance(result, list):
        head = result[:limit]
        body = "\n".join(f"  - {json.dumps(x, default=str)[:220]}" for x in head)
        more = f"\n  ... {len(result) - len(head)} more" if len(result) > len(head) else ""
        return f"{len(result)} results\n{body}{more}"
    return json.dumps(result, indent=1, default=str)[:4000]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="needle_bsky", description=__doc__)
    ap.add_argument("mode", choices=["route", "ask", "repl", "tools"])
    ap.add_argument("query", nargs="*")
    ap.add_argument("--arm", default="tuned-min", choices=list(ARMS))
    ap.add_argument("--router", default="grouped", choices=["grouped", "flat"])
    ap.add_argument("--stage1", default="heuristic", choices=["heuristic", "needle"],
                    help="grouped router only; 'needle' is the arm that measured 24pp worse")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--weights", default=None, help="a tuned .cact")
    ap.add_argument("--system", default=None, help='environment facts, e.g. "date: 2026-08-18; locale: en-US"')
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.mode == "tools":
        from .router import load_schemas

        print(json.dumps(load_schemas(a.arm), indent=1))
        return EXIT_OK

    if a.router == "grouped":
        if a.weights or a.system:
            ap.error("--weights and --system apply to the flat router; pass --router flat")
        from .grouped import GROUPS, GroupedRouter

        router = GroupedRouter(arm=a.arm, threshold=a.threshold, stage1=a.stage1)
        shape = f"{len(GROUPS)} groups of <=5"
    else:
        router = Router(arm=a.arm, threshold=a.threshold, system=a.system, weights=a.weights)
        shape = f"{len(router.schemas)} tools"

    if a.mode == "repl":
        print(f"needle_bsky [{a.arm}, {a.router}] — {shape}, gate {a.threshold}. Ctrl-D to exit.")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                return EXIT_OK
            if not q:
                continue
            d, res = router.call(q)
            print(_fmt_decision(d))
            if res is not None:
                print(_digest(res))

    query = " ".join(a.query).strip()
    if not query:
        ap.error("give me a query")

    if a.mode == "route":
        d = router.route(query)
        print(json.dumps(d.as_dict(), indent=1) if a.json else _fmt_decision(d))
        return EXIT_OK if d.tool else EXIT_REFUSED

    d, res = router.call(query)
    if a.json:
        print(json.dumps({"decision": d.as_dict(), "result": res}, indent=1, default=str))
    else:
        print(_fmt_decision(d))
        if d.decision == "escalate":
            print("  below the gate — not executed. Hand this one up.")
        elif res is not None:
            print(_digest(res))
    if d.tool is None:
        return EXIT_REFUSED
    return EXIT_OK if d.decision == "act" else EXIT_ESCALATE


if __name__ == "__main__":
    raise SystemExit(main())
