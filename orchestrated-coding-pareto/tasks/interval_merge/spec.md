# Task: interval_merge

Write a Python module defining exactly one public function:

```python
def merge(intervals: list[tuple[float, float]], join_touching: bool = True) -> list[tuple[float, float]]
```

Merge a list of closed numeric intervals `[a, b]`.

Rules:
- Each input interval is a 2-tuple `(a, b)` with `a <= b`; if any interval has `a > b`,
  raise `ValueError`. Degenerate points `(x, x)` are valid intervals.
- Two intervals **overlap** when their interiors or more than a single point intersect —
  precisely, sorting the two by start, they overlap when `second.start < first.end`.
  Overlapping intervals are always merged.
- Two intervals **touch** when `second.start == first.end` (they share exactly one
  point, e.g. `(1, 2)` and `(2, 3)`, or `(2, 2)` sitting on the end of `(1, 2)`).
  Touching intervals are merged only when `join_touching` is True.
- Merging is transitive: `(1,2),(2,3),(3,4)` with `join_touching=True` -> `[(1, 4)]`.
- The input may be in any order and may contain duplicates. The result is sorted by
  start (then end), contains plain tuples, and contains no mergeable pair.
- With `join_touching=False`, identical duplicates still collapse (`(1,2),(1,2)` ->
  `[(1,2)]`) because they overlap in more than a point... except degenerate points:
  `(2,2),(2,2)` are identical single points and also collapse to one.
  A contained interval merges too ((1,4),(2,3) -> [(1,4)]) since containment is overlap.
- Empty input returns `[]`.

No I/O. Standard library only. Only `merge` is tested.
