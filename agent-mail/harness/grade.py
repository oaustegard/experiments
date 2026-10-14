"""grade.py TREE CASES -> prints JSON {pass, total, mutates, error}"""
import copy, json, sys, traceback
tree, cases = sys.argv[1], json.load(open(sys.argv[2]))
sys.path.insert(0, tree)
res = {"pass": 0, "total": len(cases), "mutates": None, "error": None, "fails": []}
try:
    from orders.report import process
except Exception as e:
    res["error"] = f"import: {type(e).__name__}: {e}"; print(json.dumps(res)); sys.exit()
for i, c in enumerate(cases):
    try:
        got = process(c["text"], c["catalog"], copy.deepcopy(c["stock"]))
        ok = got == c["expect"]
    except Exception as e:
        got, ok = f"EXC {type(e).__name__}: {e}", False
    res["pass"] += ok
    if not ok and len(res["fails"]) < 5: res["fails"].append({"i": i, "got": str(got)[:400]})
st = {"WIDGET": 2}; before = dict(st)
try: process("A1,WIDGET,5", cases[0]["catalog"], st); res["mutates"] = st != before
except Exception: pass
print(json.dumps(res))
