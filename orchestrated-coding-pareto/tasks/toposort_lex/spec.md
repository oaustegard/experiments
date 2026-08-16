# Task: toposort_lex

Write a Python module defining a public function and an exception class:

```python
class CycleError(ValueError):
    # must expose attribute .cycle (see below)
    ...

def toposort(edges: list[tuple[str, str]], nodes: list[str] | None = None) -> list[str]
```

Topological sort returning the **lexicographically smallest** valid ordering.

Rules:
- An edge `(a, b)` means `a` must appear before `b`.
- `nodes`, when given, lists additional node names that must appear in the output even if
  they have no edges. Nodes mentioned in `edges` are always included; duplicates anywhere
  are fine and must not produce duplicate output entries.
- Among all valid topological orderings, return the lexicographically smallest one
  (compare as ordinary Python string lists — the standard "always pick the smallest
  available node next" strategy produces this).
- Duplicate edges are allowed and are equivalent to a single edge.
- Self-loops `(a, a)` are cycles.
- If the graph contains a cycle, raise `CycleError`. `CycleError` must subclass
  `ValueError`, and the raised instance must have an attribute `.cycle`: a list of
  distinct node names `[c0, c1, ..., ck]` such that every consecutive pair
  `(c0,c1), (c1,c2), ..., (ck-1,ck)` is an edge of the graph and `(ck, c0)` is also an
  edge (for a self-loop this is just `[a]`). Any actual cycle of the graph is accepted.
- The empty graph (no edges, no nodes) returns `[]`.

No I/O. Standard library only (the `heapq` module is allowed and useful).
Both `toposort` and `CycleError` are imported by the tests.
