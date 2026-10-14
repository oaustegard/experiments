def allocate(lines, stock):
    """Allocate stock to lines in input order.

    Keeps 2 units of every sku in reserve. Returns (allocs, rejected).
    Does not mutate `stock`.
    """
    remaining = {sku: max(0, units - 2) for sku, units in stock.items()}
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
