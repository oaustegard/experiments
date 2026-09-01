"""Maximal k-chord row-set types via Buneman splits, no tree census.

A tree arc a induces a split of the 2k chord endpoints X = {u_1,v_1,...,u_k,v_k};
its row is C[a,i] = [u_i in S] - [v_i in S] (S = far side; row sign is free).
Every realizable row-set is the split system of an X-tree, and every X-tree
refines to a binary tree with all 2k labels on leaves, whose split system
contains the original one. R is monotone in the row-set, so the maximal
row-set types are exactly the binary-tree split systems on 2k leaves, modulo
the group (column perm x column sign x row sign).
"""
from __future__ import annotations
import itertools, json, sys
from typing import List, Tuple, FrozenSet

def binary_trees(n: int):
    """All unrooted binary trees on leaves 0..n-1 as edge lists (leaf-insertion)."""
    # start with the 3-leaf star: internal node n, leaves 0,1,2
    next_id = [n + 1]
    def rec(edges, k):
        if k == n:
            yield list(edges); return
        for idx in range(len(edges)):
            a, b = edges[idx]
            m = next_id[0]; next_id[0] += 1
            new = edges[:idx] + edges[idx+1:] + [(a, m), (m, b), (m, k)]
            yield from rec(new, k + 1)
            next_id[0] -= 1
    yield from rec([(0, n), (1, n), (2, n)], 3)

def splits_of(edges: List[Tuple[int,int]], n: int):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    out = []
    for a, b in edges:
        # side containing b when edge removed
        seen, stack = {b}, [b]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if (x, y) in ((a, b), (b, a)): continue
                if y not in seen: seen.add(y); stack.append(y)
        out.append(frozenset(l for l in seen if l < n))
    return out

def row_of(S: FrozenSet[int], k: int) -> Tuple[int, ...]:
    # endpoint labels: u_i = 2i, v_i = 2i+1
    return tuple(int(2*i in S) - int(2*i+1 in S) for i in range(k))

def normalize_row(r):
    for x in r:
        if x: return r if x > 0 else tuple(-y for y in r)
    return r

def canonical(rows, k):
    rows = [normalize_row(r) for r in rows if any(r)]
    best = None
    for perm in itertools.permutations(range(k)):
        for signs in itertools.product((1, -1), repeat=k):
            img = tuple(sorted(set(normalize_row(tuple(signs[j]*r[perm[j]] for j in range(k))) for r in rows)))
            if best is None or img < best: best = img
    return best

def maximal_types(k: int):
    n = 2*k
    seen = {}
    cnt = 0
    for edges in binary_trees(n):
        cnt += 1
        rows = [row_of(S, k) for S in splits_of(edges, n)]
        key = canonical(rows, k)
        seen.setdefault(key, edges)
    types = list(seen.keys())
    # maximality: drop types whose row-set is a subset of another's
    maximal = [t for t in types if not any(set(t) < set(o) for o in types if o is not t)]
    return cnt, types, maximal

if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cnt, types, maximal = maximal_types(k)
    print(f"k={k}: {cnt} binary trees on {2*k} leaves, {len(types)} distinct row-set types, {len(maximal)} maximal")
    for t in maximal:
        print(len(t), t)
    json.dump({"k": k, "trees": cnt, "types": [list(map(list, t)) for t in types],
               "maximal": [list(map(list, t)) for t in maximal]},
              open(f"types_k{k}.json", "w"), indent=1)
