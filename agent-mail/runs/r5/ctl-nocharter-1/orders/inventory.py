def allocate(lines, stock):
    """Allocate stock to accepted order lines in ascending line-number order.

    lines: list of dicts with keys n, order, sku, qty.
    stock: dict sku -> units on hand. Not mutated.
    Returns (allocated, rejected):
      allocated: list of dicts (n, order, sku, qty, filled, backorder), in line order.
      rejected: list of (n, reason) tuples; reason is 'unknown sku'.
    """
    remaining = dict(stock)
    allocated = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
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
