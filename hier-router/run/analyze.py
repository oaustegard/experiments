"""CPU part: routes, accuracy, cost and unseen-leaf behaviour from out/eval.json.

    python3 -I run/analyze.py [--smoke]

Routing methods
  flat_long    one call, all library leaves plus `none`; argmax.
  flat_shard   leaves in chunks of 16, one call each; every option is scored as log(p / p_none) inside its
               own chunk, so chunks share a reference; the best positive score wins, else `none`.
  <tree>_greedy  argmax at each router from the root; `none` at any router ends the route as `none`.
  <tree>_beamB   best-first over the tree: a path's score is the product of its routers' probabilities,
               the B best partial or finished paths are kept each level, and the best finished leaf
               wins. `none` is a finished answer only at the root; elsewhere its mass just lowers the path,
               which is how a misrouted path gets dropped for a sibling.
<tree> is `tree` (LoRA) or `tree_body` (every layer trained, token table frozen).

End-to-end (Typed Decisions only): a case's five decisions are answered by the family sub-spike the route
picked (adapter-library R-sub-*, out/probs/R-sub-<family>.json), counted wrong when the route left the
case's workflow or said `none`. Writes results/summary.json.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree as T  # noqa: E402
from knn import auroc  # noqa: E402  (adapter-library/run)
from shared import FAM_WORKFLOW, examples, load_cases  # noqa: E402

ROOT = T.ROOT
BEAMS = (2, 3, 5)
TREES = ("tree", "tree_body", "tree_e9")


def source(leaf):
    if leaf in FAM_WORKFLOW:
        return "typed_decisions"
    if leaf in T.EVERYDAY_DESC:
        return "everyday"
    return "massive"


def greedy(calls, kids):
    node, tok, n = "root", 0, 0
    while node in kids:
        names, p, t = calls[node]
        tok, n = tok + t, n + 1
        node = names[int(np.argmax(p))]
        if node == "none":
            break
    return node, tok, n


def beam(calls, kids, B):
    front, done, tok, n, seen = [(1.0, "root")], [], 0, 0, set()
    while True:
        cand = []
        for s, node in front:
            if node not in kids:
                cand.append((s, node))
                continue
            names, p, t = calls[node]
            if node not in seen:
                seen.add(node)
                tok, n = tok + t, n + 1
            for nm, pi in zip(names, p):
                if nm != "none" or node == "root":
                    cand.append((s * pi, nm))
        cand.sort(key=lambda x: -x[0])
        front = cand[:B]
        if all(node not in kids for _, node in front):
            return front[0][1], tok, n


def flat_long(f):
    names, p, t = f["long"]
    return names[int(np.argmax(p))], t, 1


def flat_shard(f):
    best, arg, tok = 0.0, "none", 0
    for names, p, t in f["shards"]:
        tok += t
        pn = max(p[-1], 1e-9)
        for nm, pi in zip(names[:-1], p[:-1]):
            sc = math.log(max(pi, 1e-9) / pn)
            if sc > best:
                best, arg = sc, nm
    return arg, tok, len(f["shards"])


def methods(cfg):
    """method -> case id -> (route, tokens, calls)."""
    out = {"flat_long": {}, "flat_shard": {}}
    for cid, _ in cfg["cases"]:
        out["flat_long"][cid] = flat_long(cfg["flat"][cid])
        out["flat_shard"][cid] = flat_shard(cfg["flat"][cid])
    for tr in TREES:
        if tr not in cfg:
            continue
        out[f"{tr}_greedy"] = {cid: greedy(cfg[tr][cid], cfg["kids"]) for cid, _ in cfg["cases"]}
        for B in BEAMS:
            out[f"{tr}_beam{B}"] = {cid: beam(cfg[tr][cid], cfg["kids"], B) for cid, _ in cfg["cases"]}
    return out


def top(leaf):
    return T.path_of(leaf)[1]


def route_stats(routes, cases):
    ok = [routes[c][0] == lf for c, lf in cases]
    per_leaf = defaultdict(list)
    by_src = defaultdict(list)
    for (c, lf), o in zip(cases, ok):
        per_leaf[lf].append(o)
        by_src[source(lf)].append(o)
    wrong = [(routes[c][0], lf) for c, lf in cases if routes[c][0] != lf]
    to_leaf = [(r, lf) for r, lf in wrong if r != "none"]
    return {
        "acc": float(np.mean(ok)),
        "macro_acc": float(np.mean([np.mean(v) for v in per_leaf.values()])),
        "by_source": {k: float(np.mean(v)) for k, v in by_src.items()},
        "none_rate": float(np.mean([routes[c][0] == "none" for c, _ in cases])),
        "misroute_same_top": float(np.mean([top(r) == top(lf) for r, lf in to_leaf])) if to_leaf else None,
        "misroute_same_parent": float(np.mean([T.PARENT[r] == T.PARENT[lf] for r, lf in to_leaf])) if to_leaf else None,
        "tokens": float(np.mean([routes[c][1] for c, _ in cases])),
        "calls": float(np.mean([routes[c][2] for c, _ in cases])),
        "n": len(cases),
    }


def end_to_end(routes, cases, sub, ex_idx, ex_td):
    hits, tot = 0, 0
    for cid, lf in cases:
        if source(lf) != "typed_decisions":
            continue
        r = routes[cid][0]
        w = FAM_WORKFLOW[lf]
        for i in ex_idx[cid]:
            tot += 1
            if r in FAM_WORKFLOW and FAM_WORKFLOW[r] == w:
                p, g = sub[r][i], ex_td[w][i]["gold"]
                hits += int(np.argmax(p) == np.argmax(g))
    return hits / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / "out" / ("smoke" if a.smoke else "")
    ev = json.loads((out_dir / "eval.json").read_text())
    if (out_dir / "eval_r2.json").exists():   # round 2: tree LoRA at 9 epochs
        r2 = json.loads((out_dir / "eval_r2.json").read_text())
        ev["train"].update(r2["train"])
        for name, c in r2["configs"].items():
            ev["configs"][name].update(c)
    sub_dir = T.AL / "out" / "probs"
    sub = {f: json.loads((sub_dir / f"R-sub-{f}.json").read_text()) for f in FAM_WORKFLOW
           if (sub_dir / f"R-sub-{f}.json").exists()}
    ex_td, ex_idx = {}, defaultdict(list)
    for w in T.WF_DESC:
        ex_td[w] = examples(load_cases(w, "test"))
        for i, e in enumerate(ex_td[w]):
            ex_idx[e["case"]].append(i)
    have_e2e = len(sub) == len(FAM_WORKFLOW)

    res = {"train": ev["train"], "configs": {}, "scaling": {}}
    for name, cfg in ev["configs"].items():
        M = methods(cfg)
        lib = set(cfg["library"])
        trained_cases = [(c, lf) for c, lf in cfg["cases"] if lf in lib and lf in T.TRAINED]
        r = {m: route_stats(M[m], trained_cases) for m in M}
        if have_e2e:
            for m in M:
                r[m]["e2e_td"] = end_to_end(M[m], trained_cases, sub, ex_idx, ex_td)
            orac = {c: (lf, 0, 0) for c, lf in trained_cases}
            r["oracle"] = {"e2e_td": end_to_end(orac, trained_cases, sub, ex_idx, ex_td)}
        held = [(c, lf) for c, lf in cfg["cases"] if lf in T.HELD_OUT]
        if name == "added":
            for m in M:
                r[m]["zero_shot_held_out"] = float(np.mean([M[m][c][0] == lf for c, lf in held]))
        if name == "absent":
            kids = cfg["kids"]
            for m in M:
                r[m]["held_out_none_rate"] = float(np.mean([M[m][c][0] == "none" for c, _ in held]))
            pn_flat = lambda c: cfg["flat"][c]["long"][1][-1]
            r["flat_long"]["gap_auroc"] = auroc(np.array([pn_flat(c) for c, _ in held]),
                                                np.array([pn_flat(c) for c, _ in trained_cases]))
            for tr in TREES:
                if tr not in cfg:
                    continue
                reach, gaps_h, gaps_s, loc = [], [], [], []
                for c, lf in held:
                    par = next(p for p in T.path_of(lf)[-2::-1] if p in kids)   # nearest surviving ancestor
                    node = "root"
                    path = ["root"]
                    while node in kids and node != par:
                        names, p, _ = cfg[tr][c][node]
                        node = names[int(np.argmax(p))]
                        path.append(node)
                    reach.append(node == par)
                    names, p, _ = cfg[tr][c][par]
                    gaps_h.append(p[-1])
                    loc.append(node == par and names[int(np.argmax(p))] == "none")
                for c, lf in trained_cases:
                    par = T.PARENT[lf]
                    if any(par == T.PARENT[h] for h in T.HELD_OUT) and par in kids:
                        gaps_s.append(cfg[tr][c][par][1][-1])
                r[f"{tr}_greedy"].update({
                    "held_out_reached_parent": float(np.mean(reach)),
                    "held_out_none_at_parent": float(np.mean(loc)),
                    "gap_auroc_at_parent": auroc(np.array(gaps_h), np.array(gaps_s)),
                })
        res["configs"][name] = r
    for name, r in res["configs"].items():
        if not name.startswith("N"):
            continue
        n = int(name[1:].split("-")[0])
        for m, s in r.items():
            for k in ("acc", "macro_acc", "tokens", "calls", "e2e_td", "misroute_same_top"):
                if s.get(k) is not None:
                    res["scaling"].setdefault(m, {}).setdefault(str(n), {}).setdefault(k, []).append(s[k])
    for m in res["scaling"]:
        for n in res["scaling"][m]:
            res["scaling"][m][n] = {k: float(np.mean(v)) for k, v in res["scaling"][m][n].items()}
    dst = ROOT / ("out/smoke" if a.smoke else "results")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(res, indent=1))
    ms = list(res["scaling"])
    print("N    " + " ".join(f"{m[:14]:>14s}" for m in ms))
    for n in sorted({n for m in ms for n in res["scaling"][m]}, key=int):
        print(f"{n:4s} " + " ".join(f"{res['scaling'][m].get(n, {}).get('acc', float('nan')):14.3f}" for m in ms))
    for k in ("absent", "added"):
        if k in res["configs"]:
            print(k, json.dumps({m: {x: v for x, v in s.items() if x not in ("by_source",)}
                                 for m, s in res["configs"][k].items() if "greedy" in m or m.startswith("flat")}, indent=None)[:1500])


if __name__ == "__main__":
    main()
