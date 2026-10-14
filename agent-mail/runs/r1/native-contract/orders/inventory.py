"""Stock allocation for accepted order lines.

Contract:
    allocate(lines:list[dict n:int order:str sku:str qty:int], stock:dict[str,int])
        -> (allocated:list[dict n:int order:str sku:str qty:int filled:int backorder:int],
            rejected:list[tuple n:int reason:str])
"""


def allocate(lines, stock):
    remaining = dict(stock)  # copy: the caller's stock dict is never mutated
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
