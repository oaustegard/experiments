"""Stock allocation for accepted order lines."""


def allocate(lines, stock):
    """Allocate stock to order lines in ascending line-number order.

    lines: list of dict with keys n:int, order:str, sku:str, qty:int
    stock: dict sku -> int (units on hand); not mutated

    Returns (allocs, rejected):
      allocs:   list of dict n, order, sku, qty, filled, backorder
      rejected: list of (n:int, reason:str) for lines whose sku is not in stock
    """
    remaining = dict(stock)
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
        n, order, sku, qty = line["n"], line["order"], line["sku"], line["qty"]
        if sku not in remaining:
            rejected.append((n, "unknown sku"))
            continue
        filled = min(qty, remaining[sku])
        backorder = qty - filled
        remaining[sku] -= filled
        allocs.append({
            "n": n,
            "order": order,
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": backorder,
        })
    return allocs, rejected
