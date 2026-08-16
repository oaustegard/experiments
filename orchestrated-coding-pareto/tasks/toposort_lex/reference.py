import heapq
from collections import defaultdict


class CycleError(ValueError):
    def __init__(self, cycle):
        super().__init__(f"cycle detected: {cycle}")
        self.cycle = cycle


def _find_cycle(adj, nodes):
    color = {n: 0 for n in nodes}  # 0 white, 1 gray, 2 black
    stack = []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in sorted(adj[u]):
            if color[v] == 1:
                i = stack.index(v)
                return stack[i:]
            if color[v] == 0:
                found = dfs(v)
                if found:
                    return found
        stack.pop()
        color[u] = 2
        return None

    for n in sorted(nodes):
        if color[n] == 0:
            found = dfs(n)
            if found:
                return found
    return None


def toposort(edges, nodes=None):
    adj = defaultdict(set)
    indeg = defaultdict(int)
    all_nodes = set(nodes or [])
    for a, b in edges:
        all_nodes.add(a)
        all_nodes.add(b)
        if b not in adj[a]:
            adj[a].add(b)
            indeg[b] += 1
    heap = [n for n in all_nodes if indeg[n] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        n = heapq.heappop(heap)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    if len(out) != len(all_nodes):
        cycle = _find_cycle(adj, all_nodes)
        raise CycleError(cycle)
    return out
