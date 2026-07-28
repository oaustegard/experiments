"""Refutation of the supporting lemma in Exoo, "Three Graphs and the
Erdos-Gyarfas Conjecture" (arXiv:1403.5636), section 4.

The paper justifies G450 having no 32-cycle with:

    "any 8-cycle in Tutte-Coxeter contains (at least) two consecutive
     edges on the outer Hamiltonian cycle, and therefore at least one
     v-w path in a copy of H_15."

This script enumerates all 8-cycles of the Tutte-Coxeter graph and tests
that claim.  Result: 90 distinct 8-cycles, of which 10 use no two
consecutive outer-Hamiltonian edges.  The lemma is false, so the stated
argument for f(5) <= 450 has a gap.

Whether the CONCLUSION also fails is a separate question, and is not
settled here: a reconstruction of G450 from the paper's TikZ source does
contain a 32-cycle projecting onto one of these ten, but that depends on
the reconstruction being faithful.
"""
import networkx as nx


def tutte_coxeter():
    """Levi graph of GQ(2,2); 30 vertices, girth 8. LCF as given in the paper."""
    return nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5)


def eight_cycles(G):
    """All 8-cycles, each returned once (deduplicated by edge set)."""
    seen, out = set(), []

    def dfs(s, v, path, mask):
        if len(path) == 8:
            if G.has_edge(v, s):
                key = frozenset(frozenset((path[i], path[(i + 1) % 8])) for i in range(8))
                if key not in seen:
                    seen.add(key)
                    out.append(list(path))
            return
        for w in G[v]:
            if w > s and not (mask >> w) & 1:
                path.append(w)
                dfs(s, w, path, mask | (1 << w))
                path.pop()

    for s in G:
        dfs(s, s, [s], 1 << s)
    return out


def has_two_consecutive_outer(cycle, outer):
    """Does the cycle use two outer-Hamiltonian edges sharing a vertex?"""
    edges = [frozenset((cycle[i], cycle[(i + 1) % 8])) for i in range(8)]
    idx = [i for i, e in enumerate(edges) if e in outer]
    return any((i + 1) % 8 in idx or (i - 1) % 8 in idx for i in idx)


def main() -> None:
    G = tutte_coxeter()
    outer = {frozenset((i, (i + 1) % 30)) for i in range(30)}
    assert outer <= {frozenset(e) for e in G.edges()}, "outer Hamiltonian cycle absent"
    assert all(d == 3 for _, d in G.degree()), "not cubic"

    cycles = eight_cycles(G)
    bad = [c for c in cycles if not has_two_consecutive_outer(c, outer)]

    print(f"Tutte-Coxeter: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
    print(f"distinct 8-cycles          : {len(cycles)}")
    print(f"violating the paper's lemma: {len(bad)}  ({len(bad)/len(cycles):.1%})")
    for c in sorted(bad):
        print("   " + "-".join(f"{v:2d}" for v in c))


if __name__ == "__main__":
    main()
