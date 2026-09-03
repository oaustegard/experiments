import heapq
from collections import defaultdict
from .util import CycleError, _find_cycle


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
