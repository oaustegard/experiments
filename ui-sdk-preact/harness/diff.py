import json, sys
from collections import Counter


def load(p):
    r = {}

    def walk(s):
        for sp in s.get("specs", []):
            for t in sp["tests"]:
                res = t["results"][-1] if t["results"] else None
                st = res["status"] if res else "skipped"
                err = ""
                if res and res.get("error"):
                    err = res["error"].get("message", "")[:400]
                r[f"{sp['file']}:{sp['line']} {sp['title']}"] = (st, err)
        for c in s.get("suites", []):
            walk(c)

    for s in json.load(open(p))["suites"]:
        walk(s)
    return r


p, r = load(sys.argv[1]), load(sys.argv[2])
print("preact", Counter(v[0] for v in p.values()))
print("react ", Counter(v[0] for v in r.values()))
bad = lambda v: v[0] in ("failed", "timedOut", "interrupted")
only = [k for k in p if bad(p[k]) and not bad(r.get(k, ("passed",)))]
both = [k for k in p if bad(p[k]) and bad(r.get(k, ("passed",)))]
ronly = [k for k in r if bad(r[k]) and not bad(p.get(k, ("passed",)))]
print("preact-only", len(only), "both", len(both), "react-only", len(ronly))
for k in sorted(only):
    print(" P", p[k][0], k, "|", p[k][1].replace("\n", " ")[:300])
for k in sorted(both):
    print(" B", k)
for k in sorted(ronly):
    print(" R", k)
