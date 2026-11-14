#!/usr/bin/env python3
"""Collect every pair into results/summary.json and print a table.

Two sets: the Mac (TRM planning routes, results/bridge/*/result.json) and SwiftShader (BFS planning routes,
results/runs/sw-oracle-*.json). A pair is the same level and seed in both arms, on the same GPU."""
import json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent

def row(r):
    k = r["kills"]
    kills = int(k.split("/")[0]) if isinstance(k, str) else k
    return {"result": r["result"], "seconds": r.get("seconds", r.get("game_s")), "kills": kills, "health": r.get("health", r.get("hp")),
            "decisions": (r.get("stats") or {}).get("rollout", {}).get("decisions"), "decisionMs": (r.get("stats") or {}).get("rollout", {}).get("meanMs")}

pairs = {}
for f in sorted(glob.glob(str(HERE / "results/bridge/*/result.json"))):
    d = json.loads(Path(f).read_text())
    for r in (d.get("outputs") or {}).get("runs", []):
        pairs.setdefault(("mac-trm", r["map"], r["seed"]), {})[r["arm"]] = row(r)
for f in sorted(glob.glob(str(HERE / "results/runs/sw-oracle-*.json"))):
    r = json.loads(Path(f).read_text())
    arm = Path(f).stem.split("-")[2]
    pairs.setdefault(("swiftshader-bfs", r["map"], r["seed"]), {})[arm] = row(r)

out = {"pairs": [], "sets": {}}
for (gpu, m, s), v in sorted(pairs.items()):
    if "base" in v and "roll" in v: out["pairs"].append({"gpu": gpu, "map": m, "seed": s, **{a: v[a] for a in ("base", "roll")}})
for gpu in ["mac-trm", "swiftshader-bfs", "all"]:
    P = [p for p in out["pairs"] if gpu == "all" or p["gpu"] == gpu]
    agg = {"pairs": len(P)}
    for a in ("base", "roll"):
        agg[a] = {"deaths": sum(p[a]["result"] == "dead" for p in P), "exits": sum(p[a]["result"] in ("exit", "secret-exit") for p in P),
                  "kills": sum(p[a]["kills"] for p in P)}
    agg["pairsRollDiedBaseLived"] = sum(p["roll"]["result"] == "dead" and p["base"]["result"] != "dead" for p in P)
    agg["pairsBaseDiedRollLived"] = sum(p["base"]["result"] == "dead" and p["roll"]["result"] != "dead" for p in P)
    agg["pairsRollMoreKills"] = sum(p["roll"]["kills"] > p["base"]["kills"] for p in P)
    out["sets"][gpu] = agg
ms = [p["roll"]["decisionMs"] for p in out["pairs"] if p["gpu"] == "mac-trm" and p["roll"]["decisionMs"]]
out["sets"]["mac-trm"]["meanDecisionMs"] = round(sum(ms) / len(ms)) if ms else None
(HERE / "results/summary.json").write_text(json.dumps(out, indent=1))
for p in out["pairs"]:
    f = lambda x: f'{x["result"]:8} {x["seconds"]:>6}s k{x["kills"]:<3}'
    print(f'{p["gpu"]:16} {p["map"]} s{p["seed"]:<4} base {f(p["base"])} | roll {f(p["roll"])}')
print(json.dumps(out["sets"], indent=1))

# sign tests over the pairs, and how often the rollouts kept the bot's own move
from math import comb
def two_sided(k, n): return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
A = out["sets"]["all"]
disc = A["pairsBaseDiedRollLived"] + A["pairsRollDiedBaseLived"]
fewer = sum(p["roll"]["kills"] < p["base"]["kills"] for p in out["pairs"])
A["survivalSignP"] = float(f"{two_sided(A['pairsBaseDiedRollLived'], disc):.2g}")
A["pairsRollFewerKills"] = fewer
A["killsSignP"] = float(f"{two_sided(A['pairsRollMoreKills'], A['pairsRollMoreKills'] + fewer):.2g}")
A["pairsBothDied"] = sum(p["base"]["result"] == "dead" and p["roll"]["result"] == "dead" for p in out["pairs"])
chosen, dec = {}, 0
for f in glob.glob(str(HERE / "results/runs/sw-oracle-roll-*.json")) + glob.glob(str(HERE / "results/bridge/*/result.json")):
    d = json.loads(Path(f).read_text())
    runs = (d.get("outputs") or {}).get("runs", []) if "outputs" in d else [d]
    for r in runs:
        ro = (r.get("stats") or {}).get("rollout")
        if not ro or r.get("arm", "roll") != "roll" or ("bridge" in f and r.get("map") is None): continue
        for k, v in ro["chosen"].items(): chosen[k] = chosen.get(k, 0) + v
        dec += ro["decisions"]
out["decisions"] = {"total": dec, "keptBotShare": round(chosen.get("bot", 0) / dec, 3), "chosen": chosen}
bm = [r for f in glob.glob(str(HERE / "results/bridge/rollfight-base-*/result.json")) for r in (json.loads(Path(f).read_text()).get("outputs") or {}).get("runs", [])]
out["macBaselineAllSeeds"] = {"runs": len(bm), "deaths": sum(r["result"] == "dead" for r in bm)}
(HERE / "results/summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: A[k] for k in ("survivalSignP", "killsSignP", "pairsRollFewerKills", "pairsBothDied")}), json.dumps(out["decisions"]), json.dumps(out["macBaselineAllSeeds"]))
