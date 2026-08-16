def merge(intervals: list[tuple[float, float]], join_touching: bool = True) -> list[tuple[float, float]]:
    if not intervals:
        return []

    for a, b in intervals:
        if a > b:
            raise ValueError()

    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        current_start, current_end = current
        last_start, last_end = merged[-1]

        if current == merged[-1]:
            continue

        if current_start < last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        elif current_start == last_end and join_touching:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append(current)

    return merged
