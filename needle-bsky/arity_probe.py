#!/usr/bin/env python3
"""Isolate the arity effect on confidence, with selection and retrieval removed.

The main eval's claim — declaring an optional argument the query does not
license depresses the confidence head on an otherwise-correct call — is measured
across 18 tools, four wordings and a retrieval stage, any of which could be
carrying it. This removes all three: **one** declared tool, so the routing
decision is fixed and correct by construction, and the only thing that varies is
how many optional arguments that one schema declares.

    python3 arity_probe.py            # writes results_arity_probe.json

Three conditions per query, same tool, same wording:

    required   handle only
    +optional  handle + limit
    +noise     handle + limit + transcribe (the argument `bsky.py` really has
               and a router has no business filling)

If confidence falls as unlicensed arguments are declared, the effect is the
schema's arity and nothing else.
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE = {
    "name": "get_user_posts",
    "description": "Get the recent posts written by one named account.",
    "parameters": {
        "type": "object",
        "properties": {
            "handle": {
                "type": "string",
                "description": "a Bluesky handle such as pfrazee.com or austegard.com, with or without a leading @",
            }
        },
        "required": ["handle"],
    },
}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100, "description": "how many posts to return", "default": 20}
TRANSCRIBE = {
    "type": "string",
    "description": "If set, transcribe images that have no alt text via",
}

# Queries that name an account and license nothing else. No limit, no model.
QUERIES = [
    "what has pfrazee.com been posting lately",
    "grab danabra.mov's timeline",
    "anything new from mackuba.eu?",
    "what's simonwillison.net saying on bluesky",
    "pull @why.bsky.team's recent posts",
    "I want to read what hailey.at wrote recently",
    "show me austegard.com's posts",
    "what is retr0.id posting about",
    "recent posts from bnewbold.net",
    "read emilyliu.me's feed",
    "catch me up on nrk.no",
    "what did tapbots.com post",
    "posts by laurenshof.online",
    "let me see kongehuset.no's posts",
    "bring up futur.blue's recent activity",
    "what's on cpluspatch.com's timeline",
    "give me smallcircles.social's posts",
    "show what jay.bsky.team has written",
    "open the posts of aliceyuan.bsky.social",
    "I'd like to see nowbreezing.ntw.app's posts",
    "what has tressiemcphd.bsky.social said",
    "list the posts from ewanmorr.bsky.social",
    "check what danabra.mov wrote",
    "pull up posts by why.bsky.team",
    "read through austegard.com's recent posts",
    "surface simonwillison.net's latest",
    "what's mackuba.eu been up to",
    "posts from hailey.at please",
    "take a look at retr0.id's posts",
    "show me what nrk.no has posted",
]


def sign_test(a: list, b: list) -> dict:
    """Two-sided exact sign test on paired confidences. Ties dropped."""
    down = sum(1 for x, y in zip(a, b) if y < x)
    up = sum(1 for x, y in zip(a, b) if y > x)
    n = down + up
    k = min(down, up)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"down": down, "up": up, "n": n, "p": round(p, 5)}


def conditions() -> dict[str, dict]:
    req = copy.deepcopy(BASE)
    opt = copy.deepcopy(BASE)
    opt["parameters"]["properties"]["limit"] = LIMIT
    noise = copy.deepcopy(opt)
    noise["parameters"]["properties"]["transcribe"] = TRANSCRIBE
    return {"required": req, "+optional": opt, "+noise": noise}


def main() -> int:
    import needle

    out = {"queries": QUERIES, "conditions": {}}
    for label, schema in conditions().items():
        agent = needle.Needle(tools=[schema])
        rows = []
        for q in QUERIES:
            agent.reset()
            r = agent.complete(q)
            calls = r.get("function_calls") or []
            args = calls[0].get("arguments", {}) if calls else {}
            rows.append(
                {
                    "query": q,
                    "called": bool(calls),
                    "arguments": args,
                    "extra_args": sorted(k for k in args if k != "handle"),
                    "confidence": r.get("confidence"),
                }
            )
        conf = [x["confidence"] for x in rows if x["confidence"] is not None]
        out["conditions"][label] = {
            "declared": sorted(schema["parameters"]["properties"]),
            "mean_confidence": round(statistics.mean(conf), 4),
            "median_confidence": round(statistics.median(conf), 4),
            "called_rate": round(sum(x["called"] for x in rows) / len(rows), 3),
            "unlicensed_arg_rate": round(sum(1 for x in rows if x["extra_args"]) / len(rows), 3),
            "rows": rows,
        }
        c = out["conditions"][label]
        print(
            f"{label:10} declared {c['declared']}  mean conf {c['mean_confidence']:.4f}  "
            f"median {c['median_confidence']:.4f}  filled-unlicensed {c['unlicensed_arg_rate']:.3f}"
        )

    out["sign_tests"] = {}
    conds = list(out["conditions"])
    for i, a in enumerate(conds):
        for b in conds[i + 1 :]:
            xs = [r["confidence"] for r in out["conditions"][a]["rows"]]
            ys = [r["confidence"] for r in out["conditions"][b]["rows"]]
            t = sign_test(xs, ys)
            out["sign_tests"][f"{a}_vs_{b}"] = t
            print(f"  {a} -> {b}: {t['down']} down, {t['up']} up, p={t['p']}")

    (HERE / "results_arity_probe.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
