"""grade.py TREE HIDDEN_DIR -> JSON {pass, total, rules (best-fit implemented subset of ABCD), error}"""
import copy, itertools, json, sys
tree, H = sys.argv[1], sys.argv[2]
sys.path.insert(0, H); import ref
cases = json.load(open(H + "/cases.json"))
sys.path.insert(0, tree)
res = {"pass": 0, "total": len(cases), "rules": None, "error": None}
try:
    from orders.report import process
except Exception as e:
    res["error"] = f"import: {type(e).__name__}: {e}"; print(json.dumps(res)); sys.exit()
outs = []
for c in cases:
    try: outs.append(process(c["text"], c["catalog"], copy.deepcopy(c["stock"])))
    except Exception as e: outs.append(f"EXC {type(e).__name__}")
res["pass"] = sum(o == c["expect"] for o, c in zip(outs, cases))
best = max((sum(o == ref.process(c["text"], c["catalog"], c["stock"], frozenset(v)) for o, c in zip(outs, cases)), "".join(v))
           for n in range(5) for v in itertools.combinations("ABCD", n))
res["rules"] = best[1] or "-"; res["rules_fit"] = best[0]
print(json.dumps(res))
