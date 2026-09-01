"""Maximal k-chord row-set types, fast: unlabeled binary tree shapes on 2k leaves x perfect
matchings of the leaves into k chords. A type is such an object modulo isomorphism; the
row-set canonical form (column perm x column sign x row sign) is the dedupe key."""
import itertools, json, sys
from splits import splits_of, normalize_row, canonical

def shapes(n):
    """Unlabeled unrooted binary tree shapes with n leaves (leaf insertion + AHU dedupe)."""
    def canon(edges, n_nodes):
        adj = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        def sig(v, parent):
            return "(" + "".join(sorted(sig(w, v) for w in adj[v] if w != parent)) + ")"
        # root at every internal node, take min (n small)
        return min(sig(v, None) for v in adj if len(adj[v]) == 3) if n_nodes > 2 else "()"
    cur = [[(0, 100), (1, 100), (2, 100)]]  # leaves 0..2, internals from 100
    if n == 3:
        return cur
    k = 3
    while k < n:
        nxt, seen = [], set()
        for edges in cur:
            m_new = 100 + (k - 2)  # internals so far: 100 .. 100+k-3
            for idx in range(len(edges)):
                a, b = edges[idx]
                e2 = edges[:idx] + edges[idx+1:] + [(a, m_new), (m_new, b), (m_new, k)]
                key = canon(e2, 2 * (k + 1) - 2)
                if key not in seen:
                    seen.add(key); nxt.append(e2)
        cur = nxt; k += 1
    return cur
    k = 3
    while k < n:
        nxt, seen = [], set()
        for edges in cur:
            n_nodes = k + (k - 2)
            for idx in range(len(edges)):
                a, b = edges[idx]
                m_new = n_nodes + 1  # temporary id; relabel below
                e2 = edges[:idx] + edges[idx+1:] + [(a, m_new), (m_new, b), (m_new, k)]
                # relabel: leaves 0..k, internals k+1.. contiguous
                nodes = sorted({x for e in e2 for x in e})
                internals = [x for x in nodes if x > k]
                rl = {x: x for x in range(k + 1)}
                for j, x in enumerate(internals):
                    rl[x] = k + 1 + j
                e3 = [(rl[a_], rl[b_]) for a_, b_ in e2]
                key = canon(e3, len(nodes))
                if key not in seen:
                    seen.add(key); nxt.append(e3)
        cur = nxt; k += 1
    return cur

def matchings(items):
    if not items:
        yield []; return
    a, rest = items[0], items[1:]
    for j, b in enumerate(rest):
        for m in matchings(rest[:j] + rest[j+1:]):
            yield [(a, b)] + m

def types(k):
    n = 2 * k
    seen = {}
    cnt = 0
    for edges in shapes(n):
        S = splits_of(edges, n)  # splits over leaf ids 0..n-1
        for M in matchings(list(range(n))):
            cnt += 1
            # chord i = (u_i, v_i) = M[i]
            rows = [tuple(int(M[i][0] in s) - int(M[i][1] in s) for i in range(k)) for s in S]
            key = canonical(rows, k)
            seen.setdefault(key, (edges, M))
    ts = list(seen)
    maximal = [t for t in ts if not any(set(t) < set(o) for o in ts if o is not t)]
    return cnt, len(shapes(n)), ts, maximal

if __name__ == "__main__":
    k = int(sys.argv[1])
    cnt, ns, ts, maximal = types(k)
    print(f"k={k}: {ns} shapes, {cnt} shape x matching objects, {len(ts)} types, {len(maximal)} maximal")
    json.dump({"k": k, "objects": cnt, "types": [list(map(list, t)) for t in ts],
               "maximal": [list(map(list, t)) for t in maximal]}, open(f"types2_k{k}.json", "w"), indent=1)
