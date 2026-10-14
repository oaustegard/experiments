"""Allocate stock to accepted order lines.

Contract:
    allocate(lines:list[dict n:int order:str sku:str qty:int], stock:dict[str,int])
        -> (allocated:list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected:list[tuple n:int reason:str])
"""


def allocate(lines, stock):
    """Allocate stock to lines in ascending line-number order.

    The caller's ``stock`` dict is never mutated; a working copy is used.
    Lines whose sku is absent from ``stock`` are rejected with reason
    ``unknown sku``. ``allocated`` is returned in ascending ``n`` order.
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
