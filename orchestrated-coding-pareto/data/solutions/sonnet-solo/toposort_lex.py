"""Topological sort returning the lexicographically smallest valid ordering."""

import heapq


class CycleError(ValueError):
    """Raised when the graph contains a cycle.

    The raised instance exposes a `.cycle` attribute: a list of distinct
    node names [c0, c1, ..., ck] such that every consecutive pair
    (c0,c1), (c1,c2), ..., (ck-1,ck) is an edge of the graph and (ck, c0)
    is also an edge.
    """

    def __init__(self, cycle):
        self.cycle = cycle
        super().__init__("graph contains a cycle: {}".format(cycle))


def _find_cycle(remaining, adj):
    """Find one actual cycle within the induced subgraph on `remaining`.

    `remaining` is the set of nodes that could not be resolved by Kahn's
    algorithm; by construction the subgraph induced on them (restricted to
    edges whose endpoints are both in `remaining`) must contain a cycle.
    """
    visited = set()
    on_stack = set()
    stack = []

    def dfs(u):
        stack.append(u)
        on_stack.add(u)
        visited.add(u)
        for v in adj.get(u, []):
            if v not in remaining:
                continue
            if v in on_stack:
                idx = stack.index(v)
                return list(stack[idx:])
            if v not in visited:
                found = dfs(v)
                if found is not None:
                    return found
        stack.pop()
        on_stack.discard(u)
        return None

    for start in remaining:
        if start not in visited:
            found = dfs(start)
            if found is not None:
                return found
    # Should be unreachable: `remaining` is exactly the set of nodes left
    # over after Kahn's algorithm terminates early, which only happens
    # when a cycle exists among them.
    return list(remaining)


def toposort(edges, nodes=None):
    """Return the lexicographically smallest topological ordering.

    `edges` is a list of (a, b) pairs meaning a must appear before b.
    `nodes` optionally lists additional node names that must appear in
    the output even if they have no edges. Duplicate edges and duplicate
    node names are fine and never produce duplicate output entries.

    Raises CycleError if the graph (edges only; a self-loop counts) has
    a cycle.
    """
    edge_set = set(edges)

    all_nodes = set()
    for a, b in edge_set:
        all_nodes.add(a)
        all_nodes.add(b)
    if nodes:
        all_nodes.update(nodes)

    if not all_nodes:
        return []

    adj = {n: [] for n in all_nodes}
    indeg = {n: 0 for n in all_nodes}
    for a, b in edge_set:
        adj[a].append(b)
        indeg[b] += 1

    heap = [n for n in all_nodes if indeg[n] == 0]
    heapq.heapify(heap)

    result = []
    indeg_work = dict(indeg)
    while heap:
        u = heapq.heappop(heap)
        result.append(u)
        for v in adj[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                heapq.heappush(heap, v)

    if len(result) != len(all_nodes):
        remaining = all_nodes - set(result)
        cycle = _find_cycle(remaining, adj)
        raise CycleError(cycle)

    return result
