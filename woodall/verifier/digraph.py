"""Core capacitated-digraph verifier for Woodall / Edmonds-Giles dijoin packing.

Definitions (Feofiloff survey conventions, matching issue #163):
  - A dicut is the arc set partial(S) = {(u,v) in A : u in S, v not in S}
    for a nontrivial S (empty != S != V) with no ENTERING arcs, i.e. S is
    closed under predecessors: (u,v) in A and v in S implies u in S.
  - tau(D,u) = min over dicuts of sum of capacities of arcs in the dicut.
  - A dijoin is an arc set J that meets every dicut (equivalently: after
    contracting J the digraph is strongly connected).
  - nu(D,u) = max number of pairwise arc-disjoint dijoins, respecting arc
    capacities as an upper bound on how many packed dijoins may use that arc
    (capacity-0 "null" arcs can never be used by any packed dijoin).

n is small throughout this search (<=20 by design, per issue #163), so tau
and dicut enumeration use brute-force closed-set enumeration rather than a
max-flow reduction -- simpler to get right, and correctness (not speed) is
the point of a calibration-gated verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class Digraph:
    vertices: list
    arcs: list  # list of (u, v, label, capacity)

    def __post_init__(self):
        self.vindex = {v: i for i, v in enumerate(self.vertices)}
        self.n = len(self.vertices)
        self.arc_labels = [a[2] for a in self.arcs]
        self.arc_of = {a[2]: a for a in self.arcs}

    def out_arcs(self, v):
        return [a for a in self.arcs if a[0] == v]

    def in_arcs(self, v):
        return [a for a in self.arcs if a[1] == v]

    def is_dag(self) -> bool:
        # Kahn's algorithm
        indeg = {v: 0 for v in self.vertices}
        for u, v, *_ in self.arcs:
            indeg[v] += 1
        queue = [v for v in self.vertices if indeg[v] == 0]
        seen = 0
        while queue:
            v = queue.pop()
            seen += 1
            for u2, v2, *_ in self.out_arcs(v):
                indeg[v2] -= 1
                if indeg[v2] == 0:
                    queue.append(v2)
        return seen == self.n

    def closed_sets(self):
        """Yield all nontrivial predecessor-closed vertex sets S as frozensets.

        S is predecessor-closed iff for every arc (u,v) with v in S, u in S.
        Brute force over 2^n subsets -- fine for n <= ~20.
        """
        verts = self.vertices
        n = self.n
        for mask in range(1, (1 << n) - 1):
            S = {verts[i] for i in range(n) if mask & (1 << i)}
            ok = True
            for u, v, *_ in self.arcs:
                if v in S and u not in S:
                    ok = False
                    break
            if ok:
                yield frozenset(S)

    def dicut(self, S):
        """Arc set leaving S (assumes S is predecessor-closed)."""
        return [a for a in self.arcs if a[0] in S and a[1] not in S]

    def all_dicuts(self):
        """All distinct dicuts (as frozensets of arc labels), deduped."""
        seen = set()
        result = []
        for S in self.closed_sets():
            d = self.dicut(S)
            labels = frozenset(a[2] for a in d)
            if labels and labels not in seen:
                seen.add(labels)
                result.append((S, d))
        return result

    def tau(self):
        """Min-capacity dicut. Returns (value, witness S)."""
        best = None
        best_S = None
        for S in self.closed_sets():
            cap = sum(a[3] for a in self.dicut(S))
            if best is None or cap < best:
                best = cap
                best_S = S
        return best, best_S

    def is_dijoin(self, arc_labels) -> bool:
        """True iff arc_labels (a set of arc labels) meets every dicut."""
        labels = set(arc_labels)
        for S in self.closed_sets():
            d = self.dicut(S)
            if not any(a[2] in labels for a in d):
                return False
        return True

    def contract_and_check_strong(self, arc_labels) -> bool:
        """Alternate dijoin check: contract arc_labels, test strong connectivity.
        Used as an independent cross-check against is_dijoin (CEGAR relies on
        the dicut-based definition; this is the 'equivalently' clause from
        the issue statement, kept as a second solver-independent verifier).
        """
        # union-find over vertices, merge endpoints of contracted arcs
        parent = {v: v for v in self.vertices}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        labels = set(arc_labels)
        for u, v, lab, _ in self.arcs:
            if lab in labels:
                union(u, v)

        # build contracted adjacency (all original arcs, mapped to reps)
        adj = {}
        reps = set(find(v) for v in self.vertices)
        for r in reps:
            adj[r] = set()
        for u, v, lab, _ in self.arcs:
            ru, rv = find(u), find(v)
            if ru != rv:
                adj[ru].add(rv)

        if len(reps) <= 1:
            return True

        def reachable_from(start, adjacency):
            seen = {start}
            stack = [start]
            while stack:
                x = stack.pop()
                for y in adjacency[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            return seen

        fwd = reachable_from(next(iter(reps)), adj)
        if fwd != reps:
            return False
        radj = {r: set() for r in reps}
        for r in reps:
            for y in adj[r]:
                radj[y].add(r)
        bwd = reachable_from(next(iter(reps)), radj)
        return bwd == reps
