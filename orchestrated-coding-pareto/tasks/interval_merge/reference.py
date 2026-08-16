def merge(intervals, join_touching: bool = True):
    for a, b in intervals:
        if a > b:
            raise ValueError(f"reversed interval: ({a}, {b})")
    ivs = sorted((tuple(iv) for iv in intervals))
    out = []
    for a, b in ivs:
        if out:
            pa, pb = out[-1]
            if a < pb or (a, b) == (pa, pb) or (a == pb and (join_touching or a == b == pb == pa)):
                # overlap, exact duplicate, or touching (joined by flag or identical points)
                out[-1] = (pa, max(pb, b))
                continue
            if a == pb and not join_touching:
                # touching, not joined -- but identical degenerate duplicates collapse
                if (a, b) == (pa, pb):
                    continue
                out.append((a, b))
                continue
        out.append((a, b))
    return out
