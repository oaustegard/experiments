"""Stock allocation for accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to order lines.

    lines: iterable of dicts with at least line, sku, qty (and order).
    stock: {sku: int}. Not mutated.

    Returns (allocated, rejections):
      allocated: list of copies of the input dicts, in ascending line order,
        each with filled and backorder added.
      rejections: list of {"line": int, "reason": "unknown sku"}.
    """
    remaining = dict(stock)
    allocated = []
    rejections = []
    for entry in sorted(lines, key=lambda l: l["line"]):
        sku = entry["sku"]
        qty = entry["qty"]
        if sku not in remaining:
            rejections.append({"line": entry["line"], "reason": "unknown sku"})
            continue
        filled = min(qty, remaining[sku])
        backorder = qty - filled
        remaining[sku] -= filled
        row = dict(entry)
        row["filled"] = filled
        row["backorder"] = backorder
        allocated.append(row)
    return allocated, rejections
