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
                return stack[i + 1:]
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
