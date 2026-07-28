import sys
from networkx import from_graph6_bytes

def circumference(adj, n):
    """Longest simple cycle length via DFS from each start (start = smallest vertex on cycle)."""
    best = 0
    for s in range(n):
        # path DFS; only visit vertices > s so s is the minimum on the cycle
        stack = [(s, 1 << s, 1)]
        # iterative DFS over paths
        def dfs(v, seen, ln):
            nonlocal best
            for w in adj[v]:
                if w == s:
                    if ln >= 3 and ln > best:
                        best = ln
                elif w > s and not (seen >> w) & 1:
                    dfs(w, seen | (1 << w), ln + 1)
        dfs(s, 1 << s, 1)
    return best

sys.setrecursionlimit(10000)
hits = 0; total = 0
for line in sys.stdin.buffer:
    line = line.strip()
    if not line: continue
    G = from_graph6_bytes(line)
    n = G.number_of_nodes()
    adj = [sorted(G[v]) for v in range(n)]
    total += 1
    c = circumference(adj, n)
    if c <= 7:
        hits += 1
        print("WINDOW-HIT", line.decode(), "circ =", c)
print(f"n_graphs={total} circumference<=7: {hits}", file=sys.stderr)
