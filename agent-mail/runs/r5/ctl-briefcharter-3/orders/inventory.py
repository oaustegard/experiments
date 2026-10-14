"""Allocate stock to accepted order lines."""

RESERVE = 2


def allocate(lines, stock):
    """Allocate stock to lines in input order.

    lines: list of dict with n, order, sku, qty.
    stock: dict of sku -> units on hand (not mutated).
    Returns (allocated, rejected).
    """
    remaining = dict(stock)
    allocated = []
    rejected = []
    for line in sorted(lines, key=lambda ln: ln["n"]):
        sku = line["sku"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        qty = line["qty"]
        available = max(0, remaining[sku] - RESERVE)
        filled = min(qty, available)
        backorder = qty - filled
        remaining[sku] -= filled
        allocated.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": backorder,
        })
    return allocated, rejected
