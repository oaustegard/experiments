"""Allocate stock to accepted order lines.

Contract:
    allocate(lines: list[dict n:int order:str sku:str qty:int],
             stock: dict[str, int])
        -> (allocs: list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected: list[tuple n:int reason:str])

Lines are processed in the order given. Unknown skus are rejected with
reason 'unknown sku'. Two units of every sku are held in reserve and
never allocated. The caller's stock dict is never mutated.
"""

RESERVE = 2


def allocate(lines, stock):
    remaining = dict(stock)
    allocs = []
    rejected = []
    for line in lines:
        sku = line["sku"]
        qty = line["qty"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        # Keep RESERVE units of every sku back: only (stock - RESERVE), never below 0, is allocatable.
        available = max(0, remaining[sku] - RESERVE)
        filled = min(qty, available)
        backorder = qty - filled
        remaining[sku] -= filled
        allocs.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": backorder,
        })
    return allocs, rejected
