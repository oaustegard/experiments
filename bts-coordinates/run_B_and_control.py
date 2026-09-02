#!/usr/bin/env python3
"""Arm B (growing coordinates) + the paraphrase adversarial control."""
import json, copy
import numpy as np
import embed, arms
from queries import CASES

PARA = [
 "Worst-Case Rounding Error for Systems Whose Square Minors All Equal 0, +1, or -1",
 "Approximating Continuous Points by Binary Counterparts in Programs with Always-Integral Polyhedra",
 "Fractional Vectors and Their Whole-Number Substitutes When Column Subsets Split into Near-Balanced Halves",
]

R2 = {
 "P1": ["unsplittable flow with congestion exceeding the fractional flow by less than the maximum demand; Dinitz-Garg-Goemans",
  "conservation-preserving integralization of a fractional flow via flow decomposition and cycle cancelling",
  "ring loading and demand routing with additive maximum-demand slack",
  "origin-destination demand matrices with forced unique routing on a tree; per-edge cut load",
  "discrepancy of tree-path hypergraphs and union-of-paths set systems",
  "additive cut-preserving sparsification and integral reweighting of graph cuts",
  "tree-metric transportation: signed flux across edges and earth mover's distance on trees",
  "multilateral netting of directed obligations; discretizing bilateral settlement amounts in clearing networks"],
 "P2": ["exactly-two-ambiguous acyclic automaton; path multiplicity of a DAG recognizer",
  "reconvergent-fanout-free circuit; diamond-free and theta-free directed netlist",
  "pedigree graph rooted at a single founder; in-degree-two parental DAG",
  "derivation DAG with exactly two proofs per conclusion; uniqueness of normal form",
  "causal mediation with a single collider; path-analytic identification in a structural equation model",
  "prime event structure with binary conflict; prefix-closed configuration axioms",
  "local degree conditions forcing an out-arborescence; realizability of an abstract path system as a graph",
  "why-provenance lineage graph with two derivations per output tuple; semiring annotation"],
}
R3 = {
 "P1": ["Baranyai's rounding lemma; integral flow arguments for hypergraph decomposition",
  "Hoffman's circulation theorem; integral feasible circulations under cut lower and upper bounds",
  "balanced and Eulerian orientations, T-joins and parity arguments for unit cut imbalance",
  "hierarchical packet fair queueing; GPS fluid-to-packetized service lag bounded by the largest packet",
  "Tijdeman's chairman assignment problem; bounded-deviation sequencing of weighted agents",
  "aggregation-consistent integer reconciliation of hierarchical time series",
  "multi-echelon distribution and shipment lot-size discretization on a supply tree",
  "transportation polytopes on tree-structured supply networks; northwest-corner rule and integral extreme points"],
 "P2": ["geodetic graph and unique-path property; graphs with exactly two shortest paths per pair",
  "cactus graph and edge-disjoint cycle structure; theta-free graph characterization",
  "double occurrence word and chord diagram interlacement; terminals visited twice",
  "shared packed parse forest of ambiguity degree two; inherently ambiguous grammar",
  "staged tree and chain event graph; event-tree colouring for probability models",
  "1+1 path protection with primary and backup route per destination; disjoint path pair design",
  "anastomosis in a branching biological network; vascular and fungal lineage fusion",
  "recombining binomial lattice versus non-recombining tree; path-dependence of order two"],
}

def measure(pool, feats_stages, tidx, case_text, sig):
    D = embed.encode([arms.doc_text(p, "title") for p in pool])
    out = {}
    out["A_rank"] = arms.rank_of(D @ embed.encode([case_text], is_query=True)[0], tidx)
    out["N_rank"] = arms.rank_of(D @ embed.encode([sig], is_query=True)[0], tidx)
    stages = []
    for st in feats_stages:
        F = embed.encode(st, is_query=True)
        stages.append(D @ F.T)
    Phi1 = stages[0]
    per = [arms.rank_of(Phi1[:, f], tidx) for f in range(Phi1.shape[1])]
    out["mean_axes_rank"] = arms.rank_of(Phi1.mean(1), tidx)
    out["max_axes_rank"] = arms.rank_of(Phi1.max(1), tidx)
    out["best_single_axis_rank"] = int(min(per))
    cum = [np.hstack(stages[:k + 1]) for k in range(len(stages))]
    out["armC_reads"], _ = arms.run_sequential([cum[0]], tidx, budget=200, grow=False)
    out["armB_reads"], tr = arms.run_sequential(cum, tidx, budget=200, grow=True)
    out["armB_trace"] = [e for e in tr if e["event"] != "batch_all_negative"][:6]
    allF = cum[-1]
    perA = [arms.rank_of(allF[:, f], tidx) for f in range(allF.shape[1])]
    out["max_axes_rank_allstages"] = arms.rank_of(allF.max(1), tidx)
    out["best_single_axis_allstages"] = int(min(perA))
    return out

res = {}
for case, pn, tid in [("P1_PRE", "P1", "DOI:10.1007/s00493-004-0007-x"),
                      ("P2", "P2", "arXiv:2412.05182")]:
    pool = json.load(open(f"cache/pool_{pn}.json"))["pool"]
    sig = json.load(open(f"signatures/{pn}.json"))
    stages = [sig["features_round1"], R2[pn], R3[pn]]
    tidx = next(i for i, p in enumerate(pool) if p["id"] == tid)
    res[case] = measure(pool, stages, tidx, CASES[case]["text"], sig["signature"])
    print(case, json.dumps({k: v for k, v in res[case].items() if k != "armB_trace"}), flush=True)

# adversarial control: paraphrase the P1 target's title, everything else identical
pool = json.load(open("cache/pool_P1.json"))["pool"]
sig = json.load(open("signatures/P1.json"))
stages = [sig["features_round1"], R2["P1"], R3["P1"]]
tidx = next(i for i, p in enumerate(pool) if p["id"].startswith("DOI:"))
for k, para in enumerate(PARA, 1):
    pp = copy.deepcopy(pool)
    pp[tidx]["title"] = para
    r = measure(pp, stages, tidx, CASES["P1_PRE"]["text"], sig["signature"])
    res[f"P1_PARA{k}"] = r
    r["paraphrase"] = para
    print(f"P1_PARA{k}", json.dumps({x: y for x, y in r.items() if x != "armB_trace"})[:400], flush=True)

json.dump(res, open("results/arms_BC_control.json", "w"), indent=1)
