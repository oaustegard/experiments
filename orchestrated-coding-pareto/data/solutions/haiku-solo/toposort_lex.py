import heapq
from collections import defaultdict


class CycleError(ValueError):
    def __init__(self, cycle):
        self.cycle = cycle
        super().__init__(f"Cycle detected: {cycle}")


def toposort(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> list[str]:
    # Collect all nodes
    all_nodes = set()
    graph = defaultdict(set)  # adjacency list using sets to avoid duplicate edges
    in_degree = defaultdict(int)

    # Add edges
    for a, b in edges:
        all_nodes.add(a)
        all_nodes.add(b)

        # Add edge only if it's new (using set)
        if b not in graph[a]:
            graph[a].add(b)
            in_degree[b] += 1

    # Add additional nodes
    if nodes:
        for node in nodes:
            all_nodes.add(node)

    # Initialize in_degree for nodes with no incoming edges
    for node in all_nodes:
        if node not in in_degree:
            in_degree[node] = 0

    # Handle empty graph
    if not all_nodes:
        return []

    # Kahn's algorithm with min-heap for lexicographic ordering
    heap = []
    for node in all_nodes:
        if in_degree[node] == 0:
            heapq.heappush(heap, node)

    result = []
    while heap:
        # Pick the lexicographically smallest node
        node = heapq.heappop(heap)
        result.append(node)

        # Process neighbors
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    # Check if all nodes were processed
    if len(result) != len(all_nodes):
        # There's a cycle - find it
        cycle = _find_cycle(graph, all_nodes)
        raise CycleError(cycle)

    return result


def _find_cycle(graph, all_nodes):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in all_nodes}

    def dfs(node, path):
        color[node] = GRAY

        for neighbor in sorted(graph[node]):
            if color[neighbor] == GRAY:
                # Back edge - found a cycle
                idx = path.index(neighbor)
                return path[idx:] + [neighbor]
            elif color[neighbor] == WHITE:
                result = dfs(neighbor, path + [neighbor])
                if result:
                    return result

        color[node] = BLACK
        return None

    for node in sorted(all_nodes):
        if color[node] == WHITE:
            result = dfs(node, [node])
            if result:
                # Remove the duplicate last element
                return result[:-1]

    return None
