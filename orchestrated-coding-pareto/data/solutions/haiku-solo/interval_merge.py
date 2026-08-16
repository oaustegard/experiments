def merge(intervals: list[tuple[float, float]], join_touching: bool = True) -> list[tuple[float, float]]:
    """
    Merge a list of closed numeric intervals.

    Args:
        intervals: List of 2-tuples (a, b) where a <= b
        join_touching: If True, merge intervals that touch at exactly one point

    Returns:
        Sorted list of merged intervals with no overlapping or touching pairs
        (depending on join_touching setting)

    Raises:
        ValueError: If any interval has a > b
    """
    if not intervals:
        return []

    processed = []
    for a, b in intervals:
        if a > b:
            raise ValueError(f"Invalid interval: ({a}, {b})")
        processed.append([a, b])

    processed.sort()

    merged = [processed[0]]

    for current in processed[1:]:
        last = merged[-1]

        if current[0] == last[0] and current[1] == last[1]:
            continue
        elif current[0] < last[1]:
            merged[-1] = [last[0], max(last[1], current[1])]
        elif current[0] == last[1] and join_touching:
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)

    return [tuple(interval) for interval in merged]
