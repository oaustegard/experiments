"""Lexicographically smallest topological sort."""

import heapq

__all__ = ["CycleError", "toposort"]


class CycleError(ValueError):
    """Raised when the graph contains a cycle.

    The instance exposes ``.cycle``: a list of distinct node names
    ``[c0, c1, ..., ck]`` where each consecutive pair is an edge and
    ``(ck, c0)`` is an edge as well.
    """

    def __init__(self, message="graph contains a cycle", cycle=None):
        super().__init__(message)
        self.cycle = list(cycle) if cycle is not None else []


def _find_cycle(all_nodes, succ):
    """Return a list of distinct nodes forming a cycle, or None."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in all_nodes}

    for start in all_nodes:
        if color[start] != WHITE:
            continue
        # Iterative DFS carrying an explicit path stack.
        stack = [(start, iter(succ.get(start, ())))]
        color[start] = GRAY
        path = [start]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if color.get(nxt, WHITE) == GRAY:
                    idx = path.index(nxt)
                    return path[idx:]
                if color.get(nxt, WHITE) == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, iter(succ.get(nxt, ()))))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                path.pop()
                stack.pop()
    return None


def toposort(edges, nodes=None):
    """Return the lexicographically smallest topological ordering.

    Args:
        edges: iterable of ``(a, b)`` pairs meaning ``a`` precedes ``b``.
        nodes: optional iterable of extra node names to include.

    Raises:
        CycleError: if the graph contains a cycle.
    """
    all_nodes = []
    seen = set()

    def add_node(n):
        if n not in seen:
            seen.add(n)
            all_nodes.append(n)

    succ = {}
    indeg = {}
    edge_set = set()

    for a, b in edges:
        add_node(a)
        add_node(b)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        succ.setdefault(a, []).append(b)
        indeg[b] = indeg.get(b, 0) + 1

    if nodes is not None:
        for n in nodes:
            add_node(n)

    for n in all_nodes:
        indeg.setdefault(n, 0)

    heap = [n for n in all_nodes if indeg[n] == 0]
    heapq.heapify(heap)

    result = []
    while heap:
        node = heapq.heappop(heap)
        result.append(node)
        for nxt in succ.get(node, ()):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, nxt)

    if len(result) != len(all_nodes):
        cycle = _find_cycle(all_nodes, succ)
        if cycle is None:
            cycle = []
        raise CycleError("graph contains a cycle", cycle)

    return result
