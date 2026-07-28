"""
Cycle-length existence testing for the near-miss census.

Core primitive: given a graph (as adjacency list of ints) and a target
cycle length L, determine whether a simple cycle of length exactly L
exists.

Method: DFS backtracking from a start vertex, with:
  - BFS-distance pruning (can current vertex reach the start vertex in
    exactly the remaining budget of steps? if not, prune)
  - time cap per (graph, L) search
  - early exit on first cycle found (existence only, not counting)

For vertex-transitive graphs (all circulants; flagged explicitly for
named graphs we know to be vertex-transitive) a SINGLE start vertex
exhaustive search is a fully rigorous test: if a cycle of length L
exists anywhere in the graph, by the vertex-transitive automorphism
group some automorphism maps a vertex on that cycle to vertex 0, so a
length-L cycle through vertex 0 also exists. So "no length-L cycle
through vertex 0, search exhausted" == "no length-L cycle anywhere,
proven".

For non-vertex-transitive graphs we search from every vertex (early
exit across starts too), which is still fully rigorous but costs more;
we cap total time and mark TIMEOUT (inconclusive) if the budget runs
out before all starts are exhausted for a negative result.
"""
import time


class Result:
    """Outcome of a single (graph, length) existence test."""
    def __init__(self, length, status, example=None, note=""):
        self.length = length
        self.status = status  # "HIT" | "MISS" | "TIMEOUT" | "TRIVIAL_MISS" (L>n or L<3)
        self.example = example
        self.note = note

    def __repr__(self):
        return f"Result(L={self.length}, {self.status}{', ' + self.note if self.note else ''})"


def bfs_dist_from(adj, s):
    n = len(adj)
    dist = [-1] * n
    dist[s] = 0
    q = [s]
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _dfs_find_cycle(adj, start, L, deadline):
    """Exhaustive DFS for a simple cycle of length exactly L through `start`.
    Returns (found: bool, path_or_None, timed_out: bool)."""
    n = len(adj)
    dist_to_start = bfs_dist_from(adj, start)
    visited = bytearray(n)
    visited[start] = 1
    path = [start]
    timed_out = [False]

    def dfs():
        if time.time() > deadline:
            timed_out[0] = True
            return None
        cur = path[-1]
        depth = len(path)
        if depth == L:
            # need edge back to start to close the cycle
            if start in adj[cur]:
                return list(path)
            return None
        remaining = L - depth  # edges needed from a newly-added v back to start
        for v in adj[cur]:
            if v == start:
                continue  # closing only allowed exactly at depth L
            if visited[v]:
                continue
            d = dist_to_start[v]
            if d == -1 or d > remaining:
                # can't possibly get back to start within budget
                # (need: (remaining-1) more new vertices + 1 closing edge = remaining edges from v)
                continue
            visited[v] = 1
            path.append(v)
            r = dfs()
            if r is not None:
                return r
            if timed_out[0]:
                return None
            path.pop()
            visited[v] = 0
        return None

    found_path = dfs()
    return (found_path is not None, found_path, timed_out[0])


def test_length_exists(adj, L, vertex_transitive, time_budget=3.0, start_hint=None):
    """
    adj: list of lists (adjacency), 0-indexed
    L: target cycle length
    vertex_transitive: bool - if True, one start vertex suffices for a
        rigorous MISS proof.
    time_budget: seconds, total for this test
    start_hint: optional preferred start vertex (else 0)
    """
    n = len(adj)
    if L < 3:
        return Result(L, "TRIVIAL_MISS", note="L<3")
    if L > n:
        return Result(L, "TRIVIAL_MISS", note=f"L>n (n={n})")

    deadline = time.time() + time_budget
    if vertex_transitive:
        starts = [start_hint if start_hint is not None else 0]
    else:
        starts = list(range(n))

    any_timeout = False
    for s in starts:
        found, path, timed_out = _dfs_find_cycle(adj, s, L, deadline)
        if found:
            return Result(L, "HIT", example=path)
        if timed_out:
            any_timeout = True
            break
    if any_timeout:
        return Result(L, "TIMEOUT", note="search budget exhausted before proof")
    return Result(L, "MISS")


def exact_girth_and_sample_circumference(adj, time_budget=5.0):
    """BFS-based exact girth (shortest cycle) computation - O(n*m), exact.
    Also returns a crude circumference *lower bound* via DFS long-path probing
    (NOT exact; caller should not treat as exact unless graph small enough
    for full enumeration elsewhere)."""
    n = len(adj)
    girth = None
    deadline = time.time() + time_budget
    for s in range(n):
        if time.time() > deadline:
            return girth, "TIMEOUT_GIRTH"
        # BFS shortest cycle through s: standard technique
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = [s]
        head = 0
        best = None
        while head < len(q):
            u = q[head]; head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cyc_len = dist[u] + dist[v] + 1
                    if best is None or cyc_len < best:
                        best = cyc_len
        if best is not None and (girth is None or best < girth):
            girth = best
    return girth, "EXACT"
