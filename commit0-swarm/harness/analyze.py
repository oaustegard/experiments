"""Final per-library table across arms -> data/summary.json and a markdown table on stdout.

    python harness/analyze.py

Solo arm: pilot p-solo (tinydb, pyjwt, voluptuous) + f-solo, with f-cont where a
continuation ran. Swarm arm: the build pass (p-swarm / f-swarm) and the final
state after fixers (p-fix, f-fix, f-fix2, f-fix3, f-fix4; build pass where no
fixer was needed). Control: c-solo. Cost is input-side dollars from costs.jsonl.
portalocker and simpy are graded best-of-3 on an idle machine (see RESULTS.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

RUNS = DATA / "runs"
IDLE = {("solo", "portalocker"): 38, ("solo", "simpy"): 149,       # best of 3, load ~0.6
        ("swarm", "portalocker"): 38, ("swarm", "simpy"): 145}


def res(run):
    f = RUNS / run / "results.jsonl"
    return {r["lib"]: r for r in map(json.loads, f.open())} if f.exists() else {}


def cost(run):
    f = RUNS / run / "costs.jsonl"
    out = {}
    if f.exists():
        for r in map(json.loads, f.open()):
            lib = r["instance_id"].split(".")[0]
            out[lib] = out.get(lib, 0) + r["usd_input_side"]
    return out


def mem(run):
    f = RUNS / run / "memorization.json"
    return json.loads(f.read_text()) if f.exists() else {}


def main():
    solo = {**res("p-solo"), **res("f-solo")}
    solo_final = {**solo, **res("f-cont")}
    build = {**res("p-swarm"), **res("f-swarm")}
    final = dict(build)
    for r in ("p-fix", "f-fix", "f-fix2", "f-fix3", "f-fix4"):
        final.update(res(r))
    sc = {}
    for r in ("p-solo", "f-solo", "f-cont"):
        for k, v in cost(r).items():
            sc[k] = sc.get(k, 0) + v
    wc = {}
    for r in ("p-swarm", "f-swarm", "p-fix", "f-fix", "f-fix2", "f-fix3", "f-fix4"):
        for k, v in cost(r).items():
            wc[k] = wc.get(k, 0) + v
    m = {**mem("p-solo"), **mem("f-solo")}
    plan = {**json.loads((RUNS / "p-swarm" / "meta.json").read_text())["plan"],
            **json.loads((RUNS / "f-swarm" / "meta.json").read_text())["plan"]}
    rows = []
    for lib in sorted(solo_final):
        n = solo_final[lib]["n_target"]
        s = IDLE.get(("solo", lib), solo_final[lib]["passed"])
        b = build.get(lib, {}).get("passed")
        f = IDLE.get(("swarm", lib), final.get(lib, {}).get("passed"))
        rows.append({"lib": lib, "tests": n, "solo": s, "solo_usd": round(sc.get(lib, 0), 3),
                     "swarm_k": plan.get(lib, {}).get("k"), "swarm_build": b, "swarm_final": f,
                     "swarm_usd": round(wc.get(lib, 0), 3),
                     "verbatim_rate": m.get(lib, {}).get("verbatim_rate")})
    ctrl = res("c-solo")
    cm, cc = mem("c-solo"), cost("c-solo")
    crow = [{"lib": l, "tests": r["n_target"], "solo": r["passed"], "solo_usd": round(cc.get(l, 0), 3),
             "verbatim_rate": cm.get(l, {}).get("verbatim_rate")} for l, r in sorted(ctrl.items())]
    (DATA / "summary.json").write_text(json.dumps({"commit0_lite": rows, "control": crow}, indent=1) + "\n")

    print("| library | tests | solo | solo $ | swarm k | swarm build | swarm final | swarm $ | verbatim |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['lib']} | {r['tests']} | {r['solo']} | {r['solo_usd']:.2f} | {r['swarm_k']} | "
              f"{r['swarm_build']} | {r['swarm_final']} | {r['swarm_usd']:.2f} | {r['verbatim_rate']:.2f} |")
    print()
    print("| control library | tests | solo | solo $ | verbatim |")
    print("|---|---|---|---|---|")
    for r in crow:
        print(f"| {r['lib']} | {r['tests']} | {r['solo']} | {r['solo_usd']:.2f} | {r['verbatim_rate']:.2f} |")
    t = lambda key: sum(r[key] for r in rows)
    print(f"\nCommit0-lite: solo {t('solo')}/{t('tests')} tests, ${t('solo_usd'):.2f}; "
          f"swarm build {sum(r['swarm_build'] or 0 for r in rows)}/{t('tests')}, final {t('swarm_final')}/{t('tests')}, ${t('swarm_usd'):.2f}")
    print(f"Control: solo {sum(r['solo'] for r in crow)}/{sum(r['tests'] for r in crow)}, ${sum(r['solo_usd'] for r in crow):.2f}")


if __name__ == "__main__":
    main()
