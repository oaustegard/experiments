"""Allocate stock to accepted order lines.

Contract:
    allocate(lines:list[dict n:int order:str sku:str qty:int], stock:dict[str,int])
        -> (allocs:list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected:list[tuple n:int reason:str])
"""

RESERVE = 2  # units of every sku kept back, never allocated (bob's rule o2)


def allocate(lines, stock):
    remaining = dict(stock)  # copy: caller's stock is never mutated
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
        sku = line["sku"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        available = max(0, remaining[sku] - RESERVE)
        filled = min(line["qty"], available)
        backorder = line["qty"] - filled
        remaining[sku] -= filled
        allocs.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "qty": line["qty"],
            "filled": filled,
            "backorder": backorder,
        })
    return allocs, rejected
