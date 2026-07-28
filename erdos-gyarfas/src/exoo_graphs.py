"""
Reconstruction of the three graphs from Geoffrey Exoo,
"Three Graphs and the Erdos-Gyarfas Conjecture," arXiv:1403.5636
(main.tex source fetched and parsed directly -- ambiguities in the
rendered PDF/tikz figures were resolved by cross-checking against the
paper's own stated distances (d(v,w)=3 in H7; d(u,v)=d(u,w)=3 and
d(v,w)=5 in H15) and stated degree sequences. Both check out exactly
against the reconstruction below -- see reasoning trail in conversation.)

H7: 7 vertices, 9 edges. Two vertex-disjoint (except endpoints) paths
between attachment vertices v and w: a length-3 "floor" v-B-C-w and a
length-4 "roof" v-E-F-G-w (F is the third attachment vertex, u, the
apex), plus two cross edges B-E and C-G tying floor to roof.
  d(v,w)=3, d(u,v)=d(u,w)=2 (paper: "distance between any pair of the
  attachment vertices is at least 2"; "distance from v to w is 3"). MATCH.
  Degree sequence: v,w,u degree 2 (attachment); B,C,E,G degree 3. MATCH
  ("three vertices of degree 2, four of degree 3").

H15: two copies of H7 joined by a new vertex. Copy1's floor-start A1
joins copy2's floor-start A2 directly; the new vertex "u" joins both
copies' floor-ends D1, D2. Copy1's apex becomes external "w", copy2's
apex becomes external "v".
  d(u,v)=d(u,w)=3, d(v,w)=5 (paper states exactly this). MATCH.
"""
import networkx as nx

_uid_counter = [0]


def build_H7(prefix):
    A, B, C, D, E, F, G = [f"{prefix}.{x}" for x in "ABCDEFG"]
    edges = [(A, E), (E, F), (F, G), (G, D), (A, B), (B, C), (C, D), (B, E), (C, G)]
    return edges, {"v": A, "w": D, "u": F}


def build_H15(prefix):
    e1, m1 = build_H7(prefix + ".c1")
    e2, m2 = build_H7(prefix + ".c2")
    new_u = f"{prefix}.U"
    connector = [(m1["v"], m2["v"]), (m1["w"], new_u), (m2["w"], new_u)]
    edges = e1 + e2 + connector
    return edges, {"u": new_u, "w": m1["u"], "v": m2["u"]}


def replace_vertices(host_edges, replaced, gadget_of, u_neighbor_of):
    """
    host_edges: list of (a,b) — original host graph edges (hashable vertex ids)
    replaced: set of host vertex ids to replace
    gadget_of: dict host_vertex -> 'H7' | 'H15'
    u_neighbor_of: dict host_vertex -> the ONE neighbor (in host_edges) whose
        edge attaches to the 'u' port of that vertex's gadget. The other
        neighbor(s) attach to 'v'/'w' in encounter order (arbitrary — both
        gadgets are symmetric under v<->w swap).
    Returns: list of edges of the final graph (gadget-vertex ids are
        f"{host_vertex}.X" strings; unreplaced host vertices keep their id).
    """
    port_vertex = {}
    gadget_edges = []
    for r in replaced:
        builder = build_H7 if gadget_of[r] == "H7" else build_H15
        edges, m = builder(str(r))
        gadget_edges.extend(edges)
        port_vertex[r] = dict(m)  # 'u','v','w' -> vertex name

    # assign v/w to the non-u neighbors, in sorted-by-str order for determinism
    port_of = {}  # (r, neighbor) -> 'u'/'v'/'w'
    neighbors = {r: [] for r in replaced}
    for a, b in host_edges:
        if a in replaced:
            neighbors[a].append(b)
        if b in replaced:
            neighbors[b].append(a)
    for r in replaced:
        nbrs = neighbors[r]
        u_n = u_neighbor_of[r]
        assert u_n in nbrs, f"u_neighbor {u_n} not actually adjacent to {r}"
        others = sorted([n for n in nbrs if n != u_n], key=str)
        assert len(others) == 2, f"vertex {r} does not have exactly 3 host edges: {nbrs}"
        port_of[(r, u_n)] = "u"
        port_of[(r, others[0])] = "v"
        port_of[(r, others[1])] = "w"

    new_edges = []
    for a, b in host_edges:
        ea = port_vertex[a][port_of[(a, b)]] if a in replaced else a
        eb = port_vertex[b][port_of[(b, a)]] if b in replaced else b
        new_edges.append((ea, eb))

    return gadget_edges + new_edges


# ----------------------------------------------------------------------
# G78: Petersen -> replace hub with K3 (=G12) -> replace 11 of 12
# vertices with H7 (T0, one of the three triangle vertices, stays put).
# ----------------------------------------------------------------------
def build_G12_and_G78():
    # 9-cycle at angles 0,40,...,320 (deg spacing), + 3 chords + triangle
    angles = list(range(0, 360, 40))  # 9 values
    O = {a: f"O{a}" for a in angles}
    T = {0: "T0", 120: "T120", 240: "T240"}

    edges = []
    # outer 9-cycle
    for i in range(len(angles)):
        edges.append((O[angles[i]], O[angles[(i + 1) % len(angles)]]))
    # chords (from the -80/+80 rotated-by-{0,120,240} construction)
    for theta in (0, 120, 240):
        a1 = (theta - 80) % 360
        a2 = (theta + 80) % 360
        edges.append((O[a1], O[a2]))
    # triangle
    edges.append((T[0], T[120]))
    edges.append((T[120], T[240]))
    edges.append((T[240], T[0]))
    # spokes
    for theta in (0, 120, 240):
        edges.append((T[theta], O[theta]))

    G12 = nx.Graph()
    G12.add_edges_from(edges)
    assert G12.number_of_nodes() == 12
    assert G12.number_of_edges() == 18
    assert all(d == 3 for _, d in G12.degree())

    # u-edge assignment, derived from the black-marker edges in the
    # graph78 tikz figure (see reasoning trail):
    u_neighbor_of = {
        O[0]: O[40],
        O[40]: O[0],
        # O80's marker in the tex source ("(80:3)--(60:1)") is the ONE
        # ambiguous marker in this figure: (60:1) is not a declared node,
        # unlike every other marker which points at a real vertex. Of the
        # 3 candidate targets for O80 (O40, O120, chord-partner O280),
        # only O120 reproduces the paper's own claim (no 4-, 8-, or
        # 16-cycle in G78) -- O280 and O40 both yield a 16-cycle.
        # Resolved via that cross-check; see conversation for the
        # 3-way empirical test.
        O[80]: O[120],
        O[120]: O[160],
        O[160]: O[120],
        O[200]: O[240],
        O[240]: O[280],
        O[280]: O[240],
        O[320]: O[0],
        T[120]: O[120],      # spoke
        T[240]: O[240],      # spoke
    }
    replaced = set(u_neighbor_of.keys())
    assert replaced == set(G12.nodes()) - {T[0]}
    gadget_of = {r: "H7" for r in replaced}

    g78_edges = replace_vertices(list(G12.edges()), replaced, gadget_of, u_neighbor_of)
    G78 = nx.Graph()
    G78.add_edges_from(g78_edges)
    return G12, G78


# ----------------------------------------------------------------------
# G420: truncated icosahedron (buckyball, C60) -> replace every vertex
# with H7, u attached to the double-bond edge.
# ----------------------------------------------------------------------
def build_C60_buckyball():
    """Truncation of the icosahedral graph: each degree-5 vertex v becomes
    a pentagon of 5 new vertices P[v,n] (one per neighbor n of v); pentagon
    edges are the 'single-bond' edges; each original icosahedron edge (u,v)
    becomes one 'double-bond' edge P[u,v]--P[v,u]."""
    Ico = nx.icosahedral_graph()
    assert all(d == 5 for _, d in Ico.degree())

    cyclic_order = {}
    for v in Ico.nodes():
        nbrs = list(Ico.neighbors(v))
        sub = Ico.subgraph(nbrs)
        assert sub.number_of_edges() == 5, "neighbor-link of icosahedron vertex must be a 5-cycle"
        # walk the 5-cycle
        order = [nbrs[0]]
        prev = None
        cur = nbrs[0]
        for _ in range(4):
            nxt = [x for x in sub.neighbors(cur) if x != prev]
            nxt = nxt[0]
            order.append(nxt)
            prev, cur = cur, nxt
        assert set(order) == set(nbrs)
        cyclic_order[v] = order

    single_bond = []
    double_bond = []
    for v in Ico.nodes():
        order = cyclic_order[v]
        for i in range(5):
            n1, n2 = order[i], order[(i + 1) % 5]
            single_bond.append((f"P{v}.{n1}", f"P{v}.{n2}"))
    for u, v in Ico.edges():
        double_bond.append((f"P{u}.{v}", f"P{v}.{u}"))

    C60 = nx.Graph()
    C60.add_edges_from(single_bond + double_bond)
    assert C60.number_of_nodes() == 60, C60.number_of_nodes()
    assert C60.number_of_edges() == 90, C60.number_of_edges()
    assert all(d == 3 for _, d in C60.degree())
    return C60, single_bond, double_bond


def build_G420():
    C60, single_bond, double_bond = build_C60_buckyball()
    double_bond_partner = {}
    for a, b in double_bond:
        double_bond_partner[a] = b
        double_bond_partner[b] = a
    replaced = set(C60.nodes())
    gadget_of = {r: "H7" for r in replaced}
    u_neighbor_of = {r: double_bond_partner[r] for r in replaced}
    g420_edges = replace_vertices(list(C60.edges()), replaced, gadget_of, u_neighbor_of)
    G420 = nx.Graph()
    G420.add_edges_from(g420_edges)
    return C60, G420


# ----------------------------------------------------------------------
# G450: Tutte-Coxeter (Levi) graph, LCF [-13,-9,7,-7,9,13]^5 on 30
# vertices -> replace every vertex with H15, u attached to the chord
# edge (the one edge NOT on the base Hamiltonian 0-1-...-29-0 cycle).
# ----------------------------------------------------------------------
def build_tutte_coxeter():
    TC = nx.LCF_graph(30, [-13, -9, 7, -7, 9, 13], 5)
    assert TC.number_of_nodes() == 30
    assert TC.number_of_edges() == 45
    assert all(d == 3 for _, d in TC.degree())
    return TC


def build_G450():
    TC = build_tutte_coxeter()
    n = 30
    chord_neighbor = {}
    for i in range(n):
        cycle_nbrs = {(i - 1) % n, (i + 1) % n}
        all_nbrs = set(TC.neighbors(i))
        chord = all_nbrs - cycle_nbrs
        assert len(chord) == 1, (i, all_nbrs, cycle_nbrs)
        chord_neighbor[i] = next(iter(chord))
    replaced = set(TC.nodes())
    gadget_of = {r: "H15" for r in replaced}
    g450_edges = replace_vertices(list(TC.edges()), replaced, gadget_of, chord_neighbor)
    G450 = nx.Graph()
    G450.add_edges_from(g450_edges)
    return TC, G450
