"""Merge closed numeric intervals, with configurable handling of touching intervals."""


def merge(intervals: list[tuple[float, float]], join_touching: bool = True) -> list[tuple[float, float]]:
    """Merge a list of closed numeric intervals [a, b].

    Each input interval is a 2-tuple (a, b) with a <= b; if any interval has
    a > b, ValueError is raised. Degenerate points (x, x) are valid.

    Two intervals overlap when, sorted by start, the second interval's start
    is strictly less than the first interval's end; overlapping intervals are
    always merged. Two intervals touch when the second interval's start
    equals the first interval's end (they share exactly one point); touching
    intervals are merged only when join_touching is True. The sole exception
    is a pair of identical degenerate points (x, x) and (x, x): these always
    collapse into one, even when join_touching is False.

    Merging is transitive. The result is sorted by start (then end), contains
    plain tuples, and contains no mergeable pair. Empty input returns [].
    """
    for a, b in intervals:
        if a > b:
            raise ValueError(f"invalid interval: start {a!r} > end {b!r}")

    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda iv: (iv[0], iv[1]))

    result: list[tuple[float, float]] = []
    cur_start, cur_end = ordered[0]

    for start, end in ordered[1:]:
        if start < cur_end:
            should_merge = True
        elif start == cur_end:
            if join_touching:
                should_merge = True
            elif cur_start == cur_end and start == end and start == cur_start:
                # Identical degenerate points always collapse, regardless of
                # join_touching.
                should_merge = True
            else:
                should_merge = False
        else:
            should_merge = False

        if should_merge:
            if end > cur_end:
                cur_end = end
        else:
            result.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    result.append((cur_start, cur_end))

    return result
