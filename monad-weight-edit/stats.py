"""Paired comparison of two run files on shared uids: exact McNemar p for correct and for loop."""
import json, sys
from math import comb
from summarize import score


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main(pa, pb, out=None):
    a, b = [{json.loads(l)["uid"]: json.loads(l) for l in open(p)} for p in (pa, pb)]
    res = {}
    for t in ("mmlu", "gsm8k", "ifeval", "all"):
        us = [u for u in a if u in b and (t == "all" or a[u]["task"] == t)]
        sa, sb = [score(a[u]) for u in us], [score(b[u]) for u in us]
        row = {"n": len(us)}
        for k in ("correct", "loop"):
            gain = sum((not x[k]) and y[k] for x, y in zip(sa, sb))
            lose = sum(x[k] and not y[k] for x, y in zip(sa, sb))
            row[k] = {"base": sum(x[k] for x in sa), "edit": sum(y[k] for y in sb),
                      "only_edit": gain, "only_base": lose, "p": mcnemar(gain, lose)}
        res[t] = row
    print(json.dumps(res, indent=1))
    if out:
        open(out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
