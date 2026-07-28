#!/usr/bin/env python3
"""Validation suite for existence.test_length_exists against ground truth
we can independently confirm (small enough for exhaustive nx.simple_cycles)."""
import networkx as nx
from census import to_adj
from existence import test_length_exists

def truth_spectrum(G):
    return set(len(c) for c in nx.simple_cycles(G))

cases = [
    ("K7,7", nx.complete_bipartite_graph(7, 7), True),
    ("K5,5", nx.complete_bipartite_graph(5, 5), True),
    ("Petersen", nx.petersen_graph(), True),
    ("Heawood", nx.heawood_graph(), True),
    ("Cubical(Q3)", nx.cubical_graph(), True),
    ("K6", nx.complete_graph(6), False),
    ("Chvatal", nx.chvatal_graph(), False),
    ("PetersenGP(5,2)", nx.generalized_petersen_graph(5, 2), True),
    ("Dodecahedral", nx.dodecahedral_graph(), True),
]

ok = True
for name, G, vt in cases:
    adj, G0 = to_adj(G)
    n = len(adj)
    truth = truth_spectrum(G0)
    for L in [4, 5, 6, 7, 8, 9, 10, 16]:
        if L > n:
            continue
        r = test_length_exists(adj, L, vt, time_budget=5.0)
        expected = L in truth
        got = (r.status == "HIT")
        mark = "OK" if got == expected else "*** MISMATCH ***"
        if got != expected:
            ok = False
        if r.status == "TIMEOUT":
            mark = "TIMEOUT(inconclusive)"
        print(f"{name:20s} L={L:3d} truth={'HIT' if expected else 'miss':4s} got={r.status:8s} {mark}")

print()
print("ALL OK" if ok else "FAILURES PRESENT -- DO NOT TRUST test_length_exists YET")
