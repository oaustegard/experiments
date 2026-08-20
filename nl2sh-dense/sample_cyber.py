#!/usr/bin/env python3
"""Extend the independent cyber eval from 38 rows to a size that can resolve 0.05.

The eval in `nl2sh-selfhist` is 38 commands, 34 of them leak-free. Every
retrieval number in issue #48 is measured on those 34, where one query is 0.029
— so a hybrid arm that beats BM25 by two queries and one that beats it by luck
look identical. This samples more commands from the same corpus under the same
tiered protocol, so `gen_nl.py` can write natural language for them and the eval
grows without changing what it measures.

The protocol, reconstructed from `cyber_sample.json`'s tier field (its own
sampling script was not committed):

* **head** — utilities in the top 50 by invocation count.
* **mid** — invoked more than once, outside the top 50.
* **tail** — invoked exactly once.

Sampling is per tier, at most one command per utility, taking the *modal*
command for that utility so the row is a real thing many people typed rather
than one participant's typo. Commands already in `cyber_sample.json` are
excluded, so the new rows extend the eval rather than replacing it, and the
union is the eval that gets reported.

Two filters, both because a row that cannot be described cannot be evaluated:
commands shorter than 4 characters or longer than 120 are dropped, and so are
any whose leading token is not documented in the chunk corpus — the retrieval
tier cannot surface a page that does not exist, and scoring against one measures
corpus coverage, which `nl2sh-selfhist` already measured separately (24.4% of
utilities, 87.7% of invocations).

    python3 sample_cyber.py --cyber <unzipped>/ --n 240 --out cyber_sample_ext.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

SELFHIST = experiment("nl2sh-selfhist")
RETRIEVAL = experiment("nl2sh-retrieval")
sys.path.insert(0, str(SELFHIST))
import corpus_probe as C  # noqa: E402

MIN_LEN, MAX_LEN = 4, 120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cyber", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, default=RETRIEVAL / "data" / "chunks.jsonl")
    ap.add_argument("--existing", type=Path, default=SELFHIST / "cyber_sample.json")
    ap.add_argument("--n", type=int, default=240, help="total rows, split evenly by tier")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", type=Path, default=HERE / "cyber_sample_ext.json")
    a = ap.parse_args()

    cmds = C.load_cyber(a.cyber)
    freq = collections.Counter(u for u in (C.utility(c) for c in cmds) if u)
    documented = {json.loads(line)["utility"] for line in a.chunks.open()}
    have = {r["cmd"] for r in json.loads(a.existing.read_text())}

    top50 = {u for u, _ in freq.most_common(50)}
    by_util: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for c in cmds:
        u = C.utility(c)
        if not u or u not in documented or c in have:
            continue
        if not (MIN_LEN <= len(c) <= MAX_LEN):
            continue
        by_util[u][c] += 1

    tiers: dict[str, list[dict]] = {"head": [], "mid": [], "tail": []}
    for u, counter in by_util.items():
        tier = "head" if u in top50 else ("tail" if freq[u] == 1 else "mid")
        cmd, _ = counter.most_common(1)[0]
        tiers[tier].append({"tier": tier, "cmd": cmd, "utility": u, "freq": freq[u]})

    rng = random.Random(a.seed)
    per = a.n // 3
    out = []
    for tier in ("head", "mid", "tail"):
        pool = sorted(tiers[tier], key=lambda r: (-r["freq"], r["cmd"]))
        rng.shuffle(pool)
        out.extend(pool[:per])
        print(f"{tier}: {len(pool)} candidates, took {min(per, len(pool))}", file=sys.stderr)
    rng.shuffle(out)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} rows to {a.out.name}; "
          f"{len(set(r['utility'] for r in out))} distinct utilities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
