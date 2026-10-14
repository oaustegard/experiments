"""Allocate stock to accepted order lines.

Contract:
    allocate(lines: list[dict n:int order:str sku:str qty:int],
             stock: dict[str, int])
        -> (allocs: list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected: list[tuple n:int reason:str])

Lines are processed in ascending line number n. A sku missing from stock is
rejected with reason "unknown sku". The caller's stock dict is never mutated.
"""


def allocate(lines, stock):
    # Rule o2 (bob): keep 2 units of every sku in reserve; only
    # max(0, stock - 2) units can be allocated.
    remaining = {sku: max(0, units - 2) for sku, units in stock.items()}
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
        sku = line["sku"]
        qty = line["qty"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
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
