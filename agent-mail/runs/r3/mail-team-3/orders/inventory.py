"""Stock allocation for accepted order lines.

Contract:
    allocate(lines:list[dict n:int order:str sku:str qty:int], stock:dict[str,int])
        -> (allocs:list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected:list[tuple n:int reason:str])
"""


def allocate(lines, stock):
    remaining = dict(stock)  # never mutate the caller's dict
    allocs = []
    rejected = []
    for line in sorted(lines, key=lambda l: l["n"]):
        sku = line["sku"]
        qty = line["qty"]
        if sku not in remaining:
            rejected.append((line["n"], "unknown sku"))
            continue
        available = max(0, remaining[sku] - 2)  # keep 2 units in reserve
        filled = min(qty, available)
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
