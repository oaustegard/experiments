RESERVE = 2  # units kept back from allocation for every sku


def allocate(lines, stock):
    """Allocate stock to order lines in input order.

    lines: list of dict with n, order, sku, qty (already merged by parser).
    stock: dict sku -> units on hand. Never mutated.

    Returns (allocated, rejected):
      allocated: list[dict n order sku qty filled backorder] in input order
      rejected:  list[tuple (n, reason)]
    """
    remaining = dict(stock)  # private copy; caller's dict is untouched
    allocated = []
    rejected = []
    for line in sorted(lines, key=lambda ln: ln["n"]):
        sku = line["sku"]
        qty = line["qty"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
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
