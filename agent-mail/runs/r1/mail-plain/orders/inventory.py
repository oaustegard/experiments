"""Stock allocation for accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to accepted order lines, in ascending line order.

    lines: list of {"line": int, "order_id": str, "sku": str, "qty": int}
    stock: {sku: int}; never mutated.

    Returns (allocations, rejects):
      allocations: copies of the input lines, each with "filled" and
                   "backorder" added, sorted by "line". Unknown-sku lines
                   are excluded.
      rejects:     [{"line": int, "reason": "unknown sku"}, ...]
    """
    remaining = dict(stock)  # work on a copy; caller's dict is untouched
    allocations = []
    rejects = []
    for entry in sorted(lines, key=lambda l: l["line"]):
        sku = entry["sku"]
        if sku not in remaining:
            rejects.append({"line": entry["line"], "reason": "unknown sku"})
            continue
        qty = entry["qty"]
        available = max(remaining[sku], 0)
        filled = min(qty, available)
        remaining[sku] = available - filled
        allocated = dict(entry)
        allocated["filled"] = filled
        allocated["backorder"] = qty - filled
        allocations.append(allocated)
    return allocations, rejects
