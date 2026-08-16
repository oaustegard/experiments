import heapq


class CycleError(ValueError):
    """Raised when a cycle is detected in the graph."""
    def __init__(self, cycle):
        self.cycle = cycle
        super().__init__(f"Cycle detected: {cycle}")


def toposort(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> list[str]:
    """
    Topological sort returning the lexicographically smallest valid ordering.

    Args:
        edges: List of (source, destination) tuples representing graph edges.
        nodes: Optional list of additional node names to include in output.

    Returns:
        Lexicographically sorted topological ordering of all nodes.

    Raises:
        CycleError: If the graph contains a cycle.
    """
    # Build the graph
    from_node_map = {}  # node -> list of nodes it points to
    in_degree = {}  # node -> in-degree count

    # Process edges
    seen_edges = set()
    for a, b in edges:
        # Skip duplicate edges
        if (a, b) in seen_edges:
            continue
        seen_edges.add((a, b))

        # Initialize nodes if not present
        if a not in in_degree:
            in_degree[a] = 0
            from_node_map[a] = []
        if b not in in_degree:
            in_degree[b] = 0
            from_node_map[b] = []

        # Add edge
        from_node_map[a].append(b)
        in_degree[b] += 1

    # Add additional nodes
    if nodes:
        for node in nodes:
            if node not in in_degree:
                in_degree[node] = 0
                from_node_map[node] = []

    # Kahn's algorithm with min-heap for lexicographic ordering
    heap = [node for node in in_degree if in_degree[node] == 0]
    heapq.heapify(heap)

    result = []

    while heap:
        # Pick the lexicographically smallest node with in-degree 0
        node = heapq.heappop(heap)
        result.append(node)

        # Process its neighbors
        for neighbor in from_node_map[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    # Check for cycles
    if len(result) < len(in_degree):
        # There's a cycle - find it
        cycle = _find_cycle(from_node_map, in_degree)
        raise CycleError(cycle)

    return result


def _find_cycle(from_node_map, in_degree):
    """Find a cycle in the graph given nodes with in_degree > 0."""
    unprocessed = [node for node, degree in in_degree.items() if degree > 0]

    if not unprocessed:
        return []

    start = unprocessed[0]

    def dfs(node, path, path_set):
        if node in path_set:
            # Found cycle
            idx = path.index(node)
            return path[idx:]

        path.append(node)
        path_set.add(node)

        for neighbor in from_node_map.get(node, []):
            result = dfs(neighbor, path, path_set)
            if result:
                return result

        path.pop()
        path_set.remove(node)
        return None

    cycle = dfs(start, [], set())
    return cycle if cycle else [start]
