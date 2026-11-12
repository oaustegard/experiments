#!/usr/bin/env python3
"""Gold labels: Opus re-rates every full-read ledger paper from the full text, blind to the ledger.

Two independent passes per paper; a third pass breaks any disagreement and the gold is the
majority (or the median of three different ratings). Checkpointed per call in cache/gold/.
Writes gold.json.
"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from statistics import median

import common as c

OUT = c.HERE / "cache" / "gold"
WORKERS = 6


def one(pid, k):
    dst = OUT / f"{pid}.p{k}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    r = c.call("opus", c.gold_system(), c.paper_block(pid), timeout=1500)
    j = c.parse_json(r["text"]) if r else None
    rec = {"pid": pid, "pass": k, "call": r, "label": j,
           "practical": (j or {}).get("practical", "").upper() or None}
    if rec["practical"] not in c.RATINGS:
        rec["practical"] = None
    c.save(dst, rec)
    print(pid, k, rec["practical"], flush=True)
    return rec


def main():
    ids = c.full_papers()
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda a: one(*a), [(p, k) for p in ids for k in (1, 2)]))
        need3 = [p for p in ids
                 if one(p, 1)["practical"] != one(p, 2)["practical"]]
        list(ex.map(lambda p: one(p, 3), need3))
    led = c.ledger()
    gold = {}
    for p in ids:
        passes = [one(p, k) for k in (1, 2, 3) if (OUT / f"{p}.p{k}.json").exists()]
        rs = [x["practical"] for x in passes if x["practical"]]
        if not rs:
            gold[p] = None
            continue
        maj = max(set(rs), key=rs.count)
        g = maj if rs.count(maj) >= 2 or len(rs) == 1 else c.RATINGS[int(median(c.ORD[r] for r in rs))]
        src = next(x for x in passes if x["practical"] == g)
        gold[p] = {"title": led[p]["title"], "first_seen": led[p]["first_seen"],
                   "practical": g, "passes": rs, "ledger": led[p]["practical"],
                   "key_numbers": src["label"].get("key_numbers"), "code_weights": src["label"].get("code_weights"),
                   "why": src["label"].get("why"), "confidence": src["label"].get("confidence")}
    c.save(c.HERE / "gold.json", gold)
    print("gold:", sum(1 for v in gold.values() if v), "of", len(ids), "| tiebreaks:", len(need3))


if __name__ == "__main__":
    if len(sys.argv) > 1:  # smoke test: one paper, one pass
        print(json.dumps(one(sys.argv[1], 1), indent=1)[:3000])
    else:
        main()
