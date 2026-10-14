"""Allocate stock to accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to lines in ascending line-number order.

    Returns (allocs, rejected):
      allocs   -- list of dict n, order, sku, qty, filled, backorder
                  (only lines whose sku is in stock, sorted by n)
      rejected -- list of (n, reason) tuples; unknown sku -> (n, 'unknown sku')

    The caller's stock dict is never mutated.
    """
    remaining = dict(stock)
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda ln: ln["n"]):
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
