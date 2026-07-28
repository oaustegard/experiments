"""
Phase 3 (issue #169): the out-tree normal form for path-closed 2-path
instances, and the reduction of Conjecture 1.3 on that class to a signed
tree-discrepancy problem with a totally unimodular constraint matrix.

Lemma 4 (NOGOS.md NG-7) says: a path-closed instance with exactly 2 s->t_i
paths per terminal is an out-tree T rooted at s whose terminals are sinks
attached by two arcs, from tree nodes u_i and v_i. Such instances are
path-closed BY CONSTRUCTION (the tree gives a unique s->w path for every
tree node w, so t_i has exactly the two paths through u_i and v_i) -- which
is the whole point: every verdict on this class is a real verdict, unlike
the abstract clause systems of gadgets.py (see NG-7's false positive).

REDUCTION (Lemma 5, derived here). Write z_i = 1 if terminal i routes via
u_i, else 0; w_i = fractional weight on the u_i path. Then:

  * terminal arc (u_i, t_i): |load - x| is d_i(1-w_i) or d_i w_i, both
    <= d_i <= d_max. Terminal arcs can NEVER violate -- but they still enter
    the max, so they DO affect the value of t(w) even though they can never
    make t(w) positive. (Caught by cross-checking against the Rybin anchor:
    a tree-arcs-only t(w) returned -11 where the true value is -9. The
    counterexample PREDICATE t(w) > 0 is identical either way, since a
    terminal arc alone can never exceed d_max; the two differ only when the
    optimum sits below the bound. Both are exposed below: t_of_w is exact
    and matches core.verify_counterexample; t_tree_of_w is the pure
    reduction surrogate.)
  * tree arc a with head-subtree S_a:
        f^P(a) - x(a) = sum_i c[a,i] * d_i * (z_i - w_i),
        c[a,i] = [u_i in S_a] - [v_i in S_a]  in {-1, 0, +1}.

c[a,i] is nonzero exactly when a lies on the tree path between u_i and v_i,
with sign given by which side. So C is the network matrix of the tree with
respect to the terminal "chords" (u_i, v_i) -- TOTALLY UNIMODULAR.

  t(w)  = min_z max_a |sum_i c[a,i] d_i (z_i - w_i)|  -  d_max
  t*    = max over w in [0,1]^k of t(w);  counterexample iff t* > 0.

LEMMA 6 (equal demands are safe -- proved, see docstring of
equal_demand_no_go): if all d_i are equal, Hoffman-Kruskal integrality of
[C; I] gives an integral z with |C(z-w)|_inf < 1, hence deviation < d_max on
every arc. NO counterexample in this class has equal demands. (This
independently explains why NG-7's equal-demand candidate could never have
been real, and it matches Morell-Skutella's own divisible-demands theorem.)
Consequence: the hunt is confined to UNEQUAL demands, and the demand scaling
is exactly what breaks the TU argument (C is TU; C*diag(d) is not).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Fr
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from core import Arc, Instance, Terminal


@dataclass
class OutTreeInstance:
    """Tree arcs (parent, child) rooted at `root`, plus terminal attachments.

    attach[i] = (u_i, v_i): the two tree nodes terminal i hangs from.
    Terminal names are generated as 't{i}'. Demands are exact Fractions.
    """
    root: str
    tree_arcs: List[Arc]
    attach: List[Tuple[str, str]]
    demands: List[Fr]

    # -- structure -----------------------------------------------------------

    def nodes(self) -> List[str]:
        ns = {self.root}
        for (u, v) in self.tree_arcs:
            ns.add(u)
            ns.add(v)
        return sorted(ns)

    def children(self) -> Dict[str, List[str]]:
        ch: Dict[str, List[str]] = {}
        for (u, v) in self.tree_arcs:
            ch.setdefault(u, []).append(v)
        return ch

    def is_valid_tree(self) -> List[str]:
        """Each non-root node has exactly one parent; root has none; all
        nodes reachable from root; attachments are distinct tree nodes."""
        problems = []
        parent: Dict[str, str] = {}
        for (u, v) in self.tree_arcs:
            if v in parent:
                problems.append(f"node {v} has two parents")
            parent[v] = u
        if self.root in parent:
            problems.append("root has a parent")
        seen = {self.root}
        frontier = [self.root]
        ch = self.children()
        while frontier:
            n = frontier.pop()
            for c in ch.get(n, []):
                if c in seen:
                    problems.append(f"cycle/reconvergence at {c}")
                    continue
                seen.add(c)
                frontier.append(c)
        for n in self.nodes():
            if n not in seen:
                problems.append(f"node {n} unreachable from root")
        for i, (u, v) in enumerate(self.attach):
            if u == v:
                problems.append(f"terminal {i}: u == v (only one path)")
            for w in (u, v):
                if w not in seen:
                    problems.append(f"terminal {i}: attach node {w} not in tree")
        return problems

    def subtree(self, head: str) -> set:
        ch = self.children()
        out, stack = {head}, [head]
        while stack:
            n = stack.pop()
            for c in ch.get(n, []):
                out.add(c)
                stack.append(c)
        return out

    def coefficient_matrix(self) -> List[List[int]]:
        """C[a][i] = [u_i in S_a] - [v_i in S_a], rows = tree arcs."""
        rows = []
        for (_, head) in self.tree_arcs:
            S = self.subtree(head)
            rows.append([int(u in S) - int(v in S) for (u, v) in self.attach])
        return rows

    # -- objective -----------------------------------------------------------

    def deviation(self, C: List[List[int]], z: Sequence[int],
                  w: Sequence[Fr]) -> Fr:
        """max_a |sum_i C[a][i] d_i (z_i - w_i)| -- the reduction's core."""
        best = Fr(0)
        for row in C:
            s = Fr(0)
            for c, d, zi, wi in zip(row, self.demands, z, w):
                if c:
                    s += c * d * (zi - wi)
            if abs(s) > best:
                best = abs(s)
        return best

    def terminal_deviation(self, z: Sequence[int], w: Sequence[Fr]) -> Fr:
        """max over terminal arcs. Both arcs of terminal i give the same
        value: d_i(1-w_i) when it routes via u_i (z_i=1), else d_i w_i."""
        return max(d * ((1 - wi) if zi else wi)
                   for d, zi, wi in zip(self.demands, z, w))

    def full_deviation(self, C: List[List[int]], z: Sequence[int],
                       w: Sequence[Fr]) -> Fr:
        return max(self.deviation(C, z, w), self.terminal_deviation(z, w))

    def t_tree_of_w(self, w: Sequence[Fr],
                    C: Optional[List[List[int]]] = None) -> Fr:
        """Reduction surrogate: tree arcs only. Lower bound on t_of_w; same
        sign whenever the true value is positive."""
        C = C if C is not None else self.coefficient_matrix()
        return min(self.deviation(C, z, w)
                   for z in product((0, 1), repeat=len(self.demands))
                   ) - max(self.demands)

    def t_of_w(self, w: Sequence[Fr], C: Optional[List[List[int]]] = None) -> Fr:
        """Exact t(w), matching core.verify_counterexample on the realized
        instance (tree arcs AND terminal arcs)."""
        C = C if C is not None else self.coefficient_matrix()
        return min(self.full_deviation(C, z, w)
                   for z in product((0, 1), repeat=len(self.demands))
                   ) - max(self.demands)

    def is_counterexample(self, w: Sequence[Fr],
                          C: Optional[List[List[int]]] = None) -> bool:
        """t(w) > 0. Terminal arcs can never exceed d_max, so this is decided
        entirely by the tree arcs -- the reduction is exact for the predicate."""
        C = C if C is not None else self.coefficient_matrix()
        dmax = max(self.demands)
        return all(self.deviation(C, z, w) > dmax
                   for z in product((0, 1), repeat=len(self.demands)))

    def best_routing(self, w: Sequence[Fr]) -> Tuple[Tuple[int, ...], Fr]:
        C = self.coefficient_matrix()
        best, arg = None, None
        for z in product((0, 1), repeat=len(self.demands)):
            dev = self.full_deviation(C, z, w)
            if best is None or dev < best:
                best, arg = dev, z
        return arg, best

    # -- cross-check path: build the explicit SSUF instance ------------------

    def to_core_instance(self) -> Instance:
        """Independent code path: materialize the actual DAG + declared path
        menus so core.verify_counterexample / check_path_closure can be run
        against the reduction's arithmetic."""
        parent: Dict[str, str] = {v: u for (u, v) in self.tree_arcs}

        def tree_path(node: str) -> Tuple[Arc, ...]:
            acc: List[Arc] = []
            cur = node
            while cur != self.root:
                p = parent[cur]
                acc.append((p, cur))
                cur = p
            return tuple(reversed(acc))

        arcs = list(self.tree_arcs)
        terms = []
        for i, ((u, v), d) in enumerate(zip(self.attach, self.demands)):
            name = f"t{i}"
            a_u, a_v = (u, name), (v, name)
            arcs.extend([a_u, a_v])
            terms.append(Terminal(name, d, [tree_path(u) + (a_u,),
                                            tree_path(v) + (a_v,)]))
        return Instance(arcs=arcs, terminals=terms)

    def split_from_w(self, w: Sequence[Fr]) -> List[List[Fr]]:
        return [[Fr(wi), 1 - Fr(wi)] for wi in w]


def equal_demand_no_go(inst: OutTreeInstance) -> bool:
    """Lemma 6: equal demands cannot yield a counterexample in this class.

    PROOF. Let C be the tree's network matrix (rows = tree arcs, columns =
    terminal chords (u_i, v_i), entries in {-1,0,1}); network matrices are
    totally unimodular, and so is [C; I]. For w in [0,1]^k, Hoffman-Kruskal
    says the polyhedron

        { zeta : floor(Cw) <= C zeta <= ceil(Cw),  0 <= zeta <= 1 }

    has integral vertices (all bounds integral), and it is nonempty since it
    contains w. Take an integral point z in it. On each row a, C z and C w
    both lie in the unit interval [floor((Cw)_a), ceil((Cw)_a)], so
    |C(z - w)|_inf < 1 (and = 0 where Cw is integral). With all demands
    equal to d = d_max, every arc deviation is d * |C(z-w)|_a < d_max. So a
    satisfying routing always exists. QED.

    Returns True when the lemma applies (all demands equal).
    """
    return len(set(inst.demands)) <= 1


# ---------------------------------------------------------------------------
# MSW25 coverage test (session 3 literature gate)
# ---------------------------------------------------------------------------

def chord_graph_edges(inst: "OutTreeInstance") -> List[Tuple[str, str]]:
    """Underlying undirected graph after suppressing terminal sinks.

    A terminal has in-degree 2 and out-degree 0, so as an undirected vertex it
    has degree 2; suppressing it replaces it by the chord {u_i, v_i}. The
    result is the tree plus one chord per terminal.
    """
    edges = [tuple(sorted(a)) for a in inst.tree_arcs]
    edges += [tuple(sorted((u, v))) for (u, v) in inst.attach]
    return edges


def is_series_parallel(edges: Sequence[Tuple[str, str]]) -> bool:
    """K4-minor-freeness via the classical SP reduction: delete degree-<=1
    vertices, suppress degree-2 vertices, drop loops/parallels. The graph is
    series-parallel iff this reduces it to nothing.
    """
    E = [tuple(sorted(e)) for e in edges if e[0] != e[1]]
    changed = True
    while changed:
        changed = False
        uniq = set(E)
        if len(uniq) != len(E):
            E = list(uniq)
            changed = True
            continue
        deg: Dict[str, int] = {}
        adj: Dict[str, List[str]] = {}
        for (u, v) in E:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        for n, d in deg.items():
            if d <= 1:
                E = [e for e in E if n not in e]
                changed = True
                break
            if d == 2:
                a, b = adj[n]
                E = [e for e in E if n not in e]
                if a != b:
                    E.append(tuple(sorted((a, b))))
                changed = True
                break
    return not E


def msw25_covered(inst: "OutTreeInstance") -> bool:
    """True if Conjecture 1.3 is already PROVED for this instance by
    Majthoub Almoghrabi-Skutella-Warode (arXiv:2412.05182, Thm 2: the tight
    two-sided d_max bound for multiflows on series-parallel digraphs).

    Suppressing the terminal sinks turns the instance into tree + chords; if
    that graph is series-parallel, MSW25 applies and no counterexample can
    exist here. Use as a search filter AND as a self-test: a positive t(w)
    on an MSW25-covered instance is a BUG, not a discovery.
    """
    return is_series_parallel(chord_graph_edges(inst))
