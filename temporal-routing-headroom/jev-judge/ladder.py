#!/usr/bin/env python3
"""Simulated Haiku -> Sonnet -> Opus ladder with Jev on the decision path, and the combination check.

Rules are fixed in RESULTS.md (Method, "Ladder simulation"). Expected solved rate and cost are
computed exactly over uniform draws from each (task, arm) pool of real attempts, tasks weighted equally.

  python3 ladder.py dev  --scores data/jev/v2 --primary Q3 --out results/v2_ladder_dev.json
  python3 ladder.py test --scores data/jev/v2 --primary Q3 --dev-config results/v2_ladder_dev.json \
                         --out results/v2_ladder_test.json
"""
import argparse
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["haiku", "sonnet-low", "opus-high"]
PREFIX_TOK = 55_000
CW_RATE = {"haiku": 0.125, "sonnet-low": 2.5, "opus-high": 5.0}  # $/MTok 5m cache write (subagent_cost.py)
JEV_COST = {"low": 0.0, "mid": 0.001, "high": 0.003}


def load(split, scores_dir, primary):
    sp = json.load(open(HERE / "split.json"))[split]
    pool = {t: {a: [] for a in ARMS} for t in sp}
    for f in glob.glob(str(HERE / "data" / "attempts" / "*.json")):
        a = json.load(open(f))
        if a.get("arm") not in ARMS or a["task"] not in sp:
            continue
        sf = Path(scores_dir) / split / f"{a['arm']}_r{a['replicate']}_{a['task']}.json"
        P = json.load(open(sf))["P"][primary]
        s = None if P is None else (1 - P if primary == "Q2" else P)
        pool[a["task"]][a["arm"]].append(dict(ok=bool(a["heldout_pass"]), c=a["cost_usd"], s=s))
    return pool


def cost_of(a, arm, model):
    return a["c"] + (PREFIX_TOK * CW_RATE[arm] / 1e6 if model == "agent" else 0.0)


def jev_ladder(pool_t, lo, hi, model, J, r=0):
    L, n, arm = pool_t[ARMS[r]], len(pool_t[ARMS[r]]), ARMS[r]
    nxt = None
    ev_ok = ev_c = 0.0
    for a in L:
        ca = cost_of(a, arm, model) + J
        if r == len(ARMS) - 1 or a["s"] > hi:
            ok, c = float(a["ok"]), ca
        elif a["s"] < lo:
            nxt = nxt or jev_ladder(pool_t, lo, hi, model, J, r + 1)
            ok, c = nxt[0], ca + nxt[1]
        else:
            ok = c = 0.0
            for b in L:
                cb = ca + cost_of(b, arm, model) + J
                if b["s"] < lo:
                    nxt = nxt or jev_ladder(pool_t, lo, hi, model, J, r + 1)
                    ok += nxt[0] / n
                    c += (cb + nxt[1]) / n
                else:
                    ok += float(b["ok"]) / n
                    c += cb / n
        ev_ok += ok / n
        ev_c += c / n
    return ev_ok, ev_c


def gated_ladder(pool_t, model, r=0):
    L, n, arm = pool_t[ARMS[r]], len(pool_t[ARMS[r]]), ARMS[r]
    nxt = None
    ev_ok = ev_c = 0.0
    for a in L:
        ca = cost_of(a, arm, model)
        if a["ok"] or r == len(ARMS) - 1:
            ok, c = float(a["ok"]), ca
        else:
            nxt = nxt or gated_ladder(pool_t, model, r + 1)
            ok, c = nxt[0], ca + nxt[1]
        ev_ok += ok / n
        ev_c += c / n
    return ev_ok, ev_c


def always(pool_t, arm, model):
    L = pool_t[arm]
    return sum(a["ok"] for a in L) / len(L), sum(cost_of(a, arm, model) for a in L) / len(L)


def policies(pool, lo, hi, model, J, tasks=None):
    tasks = tasks or sorted(pool)
    def mean(fn):
        v = [fn(pool[t]) for t in tasks]
        return [sum(x[i] for x in v) / len(v) for i in (0, 1)]
    out = {"jev_ladder": mean(lambda p: jev_ladder(p, lo, hi, model, J)),
           "heldout_gated": mean(lambda p: gated_ladder(p, model)),
           "visible_gated": mean(lambda p: always(p, "haiku", model))}
    for arm in ARMS:
        out[f"always_{arm}"] = mean(lambda p, arm=arm: always(p, arm, model))
    return out


def grid():
    g = [round(0.05 * i, 2) for i in range(1, 20)]
    return [(lo, hi) for lo in g for hi in g if hi > lo]


def sweep(pool, model, J):
    rows = []
    for lo, hi in grid():
        ok, c = policies(pool, lo, hi, model, J)["jev_ladder"]
        rows.append(dict(lo=lo, hi=hi, solved=ok, cost=c))
    return rows


def choose(rows, gated_solved):
    ok = [r for r in rows if r["solved"] >= gated_solved - 0.05]
    return min(ok, key=lambda r: (r["cost"], -r["solved"])) if ok else max(rows, key=lambda r: (r["solved"], -r["cost"]))


def boot(pool, lo, hi, model, J, reps=500, seed=0):
    rng, tasks = random.Random(seed), sorted(pool)
    acc = {}
    for _ in range(reps):
        draw = [rng.choice(tasks) for _ in tasks]
        res = policies(pool, lo, hi, model, J, draw)
        d = {"jev_solved": res["jev_ladder"][0], "jev_cost": res["jev_ladder"][1],
             "jev_vs_visible_solved": res["jev_ladder"][0] - res["visible_gated"][0],
             "jev_vs_heldout_solved": res["jev_ladder"][0] - res["heldout_gated"][0],
             "jev_cost_over_opus": res["jev_ladder"][1] / res["always_opus-high"][1]}
        for k, v in d.items():
            acc.setdefault(k, []).append(v)
    return {k: [sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v)) - 1]] for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--scores", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-config")
    a = ap.parse_args()
    pool = load(a.split, a.scores, a.primary)
    n_att = sum(len(v) for t in pool.values() for v in t.values())
    out = {"split": a.split, "primary": a.primary, "n_attempts": n_att, "tasks": sorted(pool), "models": {}}
    for model in ("headless", "agent"):
        for jl, J in JEV_COST.items():
            key = f"{model}/jev_{jl}"
            base = policies(pool, 0.5, 0.6, model, J)
            if a.split == "dev":
                rows = sweep(pool, model, J)
                cfg = choose(rows, base["heldout_gated"][0])
            else:
                cfg = json.load(open(a.dev_config))["models"][key]["chosen"]
            res = policies(pool, cfg["lo"], cfg["hi"], model, J)
            ent = {"chosen": {"lo": cfg["lo"], "hi": cfg["hi"]}, "policies": res}
            if a.split == "test":
                ent["bootstrap95"] = boot(pool, cfg["lo"], cfg["hi"], model, J)
            else:
                ent["sweep"] = rows
            out["models"][key] = ent
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for key, ent in out["models"].items():
        r = ent["policies"]
        print(f"{a.split} {key}: lo={ent['chosen']['lo']} hi={ent['chosen']['hi']}  " + "  ".join(
            f"{k}={v[0]:.2f}/${v[1]:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
