"""Score dry-run rows per (model, variant, decode):

    python3 score.py runs/r2.jsonl runs/r3.jsonl [--by model,variant,decode] [--json out.json]

finished   ended on its own (end marker or EOS) with a closed thinking block
loop       ran to the limit inside a repetition loop (common.loop_period)
named      on the identity probes (who, maker): the answer says "Monad"
leak       on probes that are not about the model: the answer talks about itself as a model
           (model, parameter, browser, Pleias, Monad, SYNTH); system text leaking into answers
words      median words in the answer
short_ok   on photo1 and summ ("one sentence"): the answer is one sentence
multi_ok   on multi (follow-up "How many people live there?"): the answer gives a number
"""
import argparse
import json
import re
import statistics
from collections import defaultdict

SELF = {"who", "maker", "feel", "web"}
LEAK = re.compile(r"\b(language model|parameters?|browser|pleias|monad|synth|small model|models like (me|myself))\b", re.I)


def sentences(t):
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def score(rows):
    n = len(rows)
    out = {"n": n,
           "finished": sum(r["stopped"] and r["closed"] for r in rows) / n,
           "loop": sum(bool(r["loop"]) for r in rows) / n,
           "words": statistics.median(len(r["answer"].split()) for r in rows),
           "tokens": statistics.median(r["tokens"] for r in rows)}
    ident = [r for r in rows if r["probe"] in ("who", "maker")]
    if ident:
        out["named"] = sum("monad" in r["answer"].lower() for r in ident) / len(ident)
    other = [r for r in rows if r["probe"] not in SELF]
    if other:
        out["leak"] = sum(bool(LEAK.search(r["answer"])) for r in other) / len(other)
    one = [r for r in rows if r["probe"] in ("photo1", "summ")]
    if one:
        out["short_ok"] = sum(r["answer"] != "" and sentences(r["answer"]) == 1 for r in one) / len(one)
    mt = [r for r in rows if r["probe"] == "multi"]
    if mt:
        out["multi_ok"] = sum(bool(re.search(r"\d", r["answer"])) for r in mt) / len(mt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", default="model,variant,decode")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.files for l in open(f)]
    keys = a.by.split(",")
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r[k]) for k in keys)].append(r)
    res = {}
    cols = ["n", "finished", "loop", "named", "leak", "short_ok", "multi_ok", "words", "tokens"]
    print(" ".join(f"{k:>8s}" for k in keys), " ".join(f"{c:>8s}" for c in cols))
    for g in sorted(groups):
        s = score(groups[g])
        res["/".join(g)] = s
        print(" ".join(f"{x:>8s}" for x in g), " ".join(
            f"{s[c]:8.2f}" if isinstance(s.get(c), float) else f"{s.get(c, ''):>8}" for c in cols))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
