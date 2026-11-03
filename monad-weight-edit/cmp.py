"""Compare two run files on the uids they share: per-task acc/loop/open."""
import json, sys
from summarize import score
a, b = [{json.loads(l)["uid"]: json.loads(l) for l in open(p)} for p in sys.argv[1:3]]
uids = [u for u in a if u in b]
for t in ("mmlu", "gsm8k", "ifeval", "all"):
    us = [u for u in uids if t == "all" or a[u]["task"] == t]
    sa, sb = [score(a[u]) for u in us], [score(b[u]) for u in us]
    f = lambda s, k: round(sum(x[k] for x in s) / len(s), 3)
    print(t, len(us), {k: (f(sa, k), f(sb, k)) for k in ("correct", "loop", "open")})
