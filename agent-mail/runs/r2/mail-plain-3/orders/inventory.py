"""Allocate stock to accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to lines in ascending line-number order.

    lines: list of {'order', 'line', 'sku', 'qty'} (accepted lines).
    stock: {sku: int}. Not mutated.

    Returns (allocated, rejected):
      allocated: each input line's dict plus 'filled' and 'backorder' (only
                 lines whose sku is known to stock).
      rejected:  [{'line': int, 'reason': 'unknown sku'}] for unknown skus.
    """
    remaining = dict(stock)
    ordered = sorted(lines, key=lambda l: l["line"])
    allocated = []
    rejected = []
    for line in ordered:
        sku = line["sku"]
        if sku not in remaining:
            rejected.append({"line": line["line"], "reason": "unknown sku"})
            continue
        qty = line["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        out = dict(line)
        out["filled"] = filled
        out["backorder"] = qty - filled
        allocated.append(out)
    return allocated, rejected
