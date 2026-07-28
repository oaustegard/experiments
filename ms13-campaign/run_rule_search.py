"""Task C driver (rescoped remit): test 4 candidate deterministic rounding
rules against all 30 k=3 row-set types, with exact Fraction arithmetic
throughout. Loads the type list from /tmp/k3_enum_progress.json (produced by
k3_proof.py's Task A enumeration -- reused here, not recomputed).
"""
import json
import time
from fractions import Fraction as Fr

import k3_proof as k

d = json.load(open("/tmp/k3_enum_progress.json"))
all_types = [tuple(tuple(r) for r in t["rows"]) for t in d["types"]]
print(f"loaded {len(all_types)} types", flush=True)

RYBIN = ((0, 1, -1), (1, -1, 0), (1, -1, 1))
assert RYBIN in all_types, "Rybin canonical type missing from enumeration!"

# Known extremal witnesses (exact) to force-check on every type. IMPORTANT:
# a witness (p,q) is only meaningful paired with the EXACT row ordering/signs
# it was derived against -- canonicalization (column permute + sign flip)
# scrambles which p_i goes with which column, so each witness below is
# recomputed fresh against ITS OWN type's canonical row tuple (as stored by
# the enumeration), never hand-copied from a different representation.
# (First version of this script hardcoded the Rybin witness from the RAW
# [(1,1,1),(1,1,0),(1,0,1)] realization and paired it with the CANONICAL
# ((0,1,-1),(1,-1,0),(1,-1,1)) rows -- caught immediately by the sanity
# assert below giving 1/4 instead of 3/4. Fixed by deriving witnesses
# in-place per type instead of copying across representations.)
rybin_w = k.find_exact_witness(list(RYBIN), [Fr(3, 4)])
RYBIN_WITNESS = ([Fr(x) for x in rybin_w["p"]], [Fr(x) for x in rybin_w["q"]])
maximal_types = [tuple(tuple(r) for r in t["rows"]) for t in d["types"]
                  if len(t["rows"]) == 6]
extra_witnesses = [RYBIN_WITNESS]
for mt in maximal_types:
    w = k.find_exact_witness(list(mt), [Fr(3, 4)])
    extra_witnesses.append(([Fr(x) for x in w["p"]], [Fr(x) for x in w["q"]]))
EXTRA_POINTS = extra_witnesses  # only valid against THEIR OWN type; see below

# sanity: Rybin witness must hit exactly 3/4 on the Rybin type itself under
# brute-force exact evaluation (independent of any rule) before we trust
# anything downstream.
val, _ = k.exact_min_over_z(list(RYBIN), *RYBIN_WITNESS)
print(f"Rybin witness sanity check: exact_min_over_z = {val} (expect 3/4)",
      flush=True)
assert val == Fr(3, 4)

results = {}
t0 = time.time()
for rule_name, rule_fn in k._RULES.items():
    print(f"\n=== rule {rule_name} ===", flush=True)
    per_type = []
    worst_overall = Fr(-1)
    worst_overall_type = None
    n_fail = 0
    for idx, rows in enumerate(all_types):
        # a witness (p,q) is only meaningful against the SPECIFIC row
        # tuple it was derived for (see note above) -- attach the matching
        # extremal witness only when this IS that type.
        pts = []
        if rows == RYBIN:
            pts.append(RYBIN_WITNESS)
        if rows in maximal_types:
            pts.append(extra_witnesses[1 + maximal_types.index(rows)])
        res = k.sweep_rule(rule_fn, list(rows), n_trials=500, seed=42,
                            extra_points=pts)
        per_type.append({"rows": list(rows), "n_rows": len(rows), **res})
        wv = Fr(res["worst_value"])
        if wv > worst_overall:
            worst_overall, worst_overall_type = wv, rows
        if res["exceeds_threshold"]:
            n_fail += 1
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(all_types)} types checked, "
                  f"{n_fail} failures so far, worst={worst_overall} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    results[rule_name] = {
        "per_type": per_type, "n_types_failed": n_fail,
        "n_types_total": len(all_types),
        "worst_overall": str(worst_overall),
        "worst_overall_type": list(worst_overall_type) if worst_overall_type else None,
        "certifies_3_4_everywhere": n_fail == 0,
    }
    print(f"  RULE {rule_name}: fails on {n_fail}/{len(all_types)} types, "
          f"worst value seen = {worst_overall}", flush=True)

with open("k3_rule_search_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\nWrote k3_rule_search_results.json ({time.time()-t0:.1f}s total)",
      flush=True)

print("\n=== SUMMARY ===")
for rule_name, r in results.items():
    print(f"{rule_name}: {'CERTIFIES 3/4' if r['certifies_3_4_everywhere'] else 'FAILS'} "
          f"(fails on {r['n_types_failed']}/{r['n_types_total']}, "
          f"worst={r['worst_overall']})")
