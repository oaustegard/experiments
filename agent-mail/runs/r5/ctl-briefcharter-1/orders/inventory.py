"""Allocate stock to accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to lines in input order.

    Returns (allocated, rejected):
      allocated: list of dicts with n, order, sku, qty, filled, backorder
                 (one per accepted line, in input order)
      rejected:  list of (n, reason) tuples for lines whose sku is unknown

    Two units of every sku are held in reserve and never allocated.
    The caller's stock dict is not mutated.
    """
    # Keep 2 units of every sku in reserve.
    remaining = {sku: max(0, units - 2) for sku, units in stock.items()}
    allocated = []
    rejected = []
    for line in sorted(lines, key=lambda line: line["n"]):
        sku = line["sku"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        qty = line["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        allocated.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": qty - filled,
        })
    return allocated, rejected
