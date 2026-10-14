"""Allocate stock to accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to lines in ascending line-number order.

    lines: list of dict with n, order, sku, qty
    stock: dict sku -> units on hand (never mutated)
    Returns (allocs, rejected):
      allocs: list of dict n, order, sku, qty, filled, backorder
      rejected: list of (n, reason) tuples
    """
    remaining = dict(stock)
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
        sku = line["sku"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        qty = line["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        allocs.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": qty - filled,
        })
    return allocs, rejected
