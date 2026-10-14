"""Allocate stock to accepted order lines.

allocate(lines, stock) -> (allocs, rejects)

lines:   accepted lines in file order, dicts with line:int, order:str, sku:str, qty:int.
stock:   {sku: int}. Never mutated.
allocs:  one dict per line that has a sku in stock. Input keys are kept and
         filled:int and backorder:int are added (filled + backorder == qty).
rejects: [{line:int, reason:'unknown sku'}] for lines whose sku is not in stock.
"""


def allocate(lines, stock):
    remaining = dict(stock)  # local copy; the caller's dict is untouched
    allocs = []
    rejects = []
    for ln in sorted(lines, key=lambda x: x["line"]):
        sku = ln["sku"]
        if sku not in remaining:
            rejects.append({"line": ln["line"], "reason": "unknown sku"})
            continue
        qty = ln["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        alloc = dict(ln)
        alloc["filled"] = filled
        alloc["backorder"] = qty - filled
        allocs.append(alloc)
    return allocs, rejects
