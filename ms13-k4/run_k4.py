"""Q7' at k=4: R per maximal row-set type. Float screen + exact rational witness."""
import json, sys, time
from fractions import Fraction as Fr
import k4_proof as m

T = json.load(open("types_k4.json"))
maximal = [tuple(tuple(r) for r in t) for t in T["maximal"]]
print(f"{len(maximal)} maximal types; {len(T['types'])} types total; {T['trees']} trees")

# completeness sanity: the campaign's k=4 refuting row-set must be dominated by some maximal type
ref = ((0,0,0,1),(0,0,1,-1),(0,1,-1,0),(1,-1,1,-1),(1,0,0,-1),(1,0,1,-1))
dom = [i for i, t in enumerate(maximal) if m.is_dominated(ref, t)]
print("campaign k=4 refuting row-set dominated by maximal types:", dom)

out = []
sel = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(len(maximal))
for i in sel:
    rows = maximal[i]
    t0 = time.time()
    f = m.float_R_milp(rows)
    w = m.find_exact_witness(rows, [Fr(4,5), Fr(3,4), Fr(2,3)])
    print(f"type {i} ({len(rows)} rows): floatR={f.get('R')}  exact witness value={w['value']}  p={w['p']} q={w['q']}  [{time.time()-t0:.1f}s]", flush=True)
    out.append({"type": i, "rows": [list(r) for r in rows], "floatR": f.get("R"),
                "witness_value": str(w["value"]), "p": w["p"], "q": w["q"]})
json.dump(out, open("k4_lower_" + (sys.argv[1].replace(",", "_") if len(sys.argv) > 1 else "all") + ".json", "w"), indent=1)
