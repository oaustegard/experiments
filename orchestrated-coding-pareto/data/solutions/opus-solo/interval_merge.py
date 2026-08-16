"""Merge closed numeric intervals.

Public API: :func:`merge`.
"""

__all__ = ["merge"]


def _validate(intervals):
    """Normalize the input into a list of ``(a, b)`` pairs.

    Raises ``ValueError`` for anything that is not a 2-element pair of
    numbers with ``a <= b``.
    """
    if intervals is None:
        raise ValueError("intervals must be an iterable of (a, b) pairs")

    normalized = []
    for item in intervals:
        try:
            a, b = item
        except (TypeError, ValueError):
            raise ValueError(
                "each interval must be a 2-tuple (a, b), got: %r" % (item,)
            )
        try:
            ok = a <= b
        except TypeError:
            raise ValueError(
                "interval endpoints must be comparable numbers, got: %r" % (item,)
            )
        if not ok:
            # Covers a > b as well as non-orderable values such as NaN.
            raise ValueError("interval start must be <= end, got: (%r, %r)" % (a, b))
        normalized.append((a, b))
    return normalized


def merge(intervals, join_touching=True):
    """Merge a list of closed intervals ``[a, b]``.

    Two intervals overlap when, sorted by start, ``second.start <
    first.end``; overlapping intervals are always merged. Two intervals
    touch when ``second.start == first.end``; touching intervals are
    merged only when ``join_touching`` is True. Identical degenerate
    points always collapse.

    Merging is transitive. The result is sorted by start (then end),
    contains plain tuples, and contains no mergeable pair.

    Raises ``ValueError`` if any interval has ``a > b`` or is malformed.
    """
    items = _validate(intervals)
    if not items:
        return []

    items.sort(key=lambda p: (p[0], p[1]))

    result = []
    cur_start, cur_end = items[0]
    for start, end in items[1:]:
        if join_touching:
            mergeable = start <= cur_end
        else:
            # Strict overlap (more than a single shared point) ...
            mergeable = start < cur_end
            if not mergeable:
                # ... plus the identical-degenerate-point case: two copies
                # of the same single point collapse to one.
                mergeable = (
                    start == cur_end and end == cur_end and cur_start == cur_end
                )
        if mergeable:
            if end > cur_end:
                cur_end = end
        else:
            result.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    result.append((cur_start, cur_end))
    return result
