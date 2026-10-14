"""Stock allocation for accepted order lines (owner: carol).

Interface
---------
allocate(lines, stock) -> (allocations, rejections)

lines : list of dict, the accepted (already merged) order lines, in any order.
        Each dict has at least:
          "line"  : int   line number in the input file
          "order" : any   order id/key the line belongs to (passed through)
          "sku"   : str
          "qty"   : int
        Other keys are passed through untouched.
stock : dict {sku: int}, units on hand. Never mutated.

Returns
-------
allocations : list of dict, one per line that was not rejected, in input
              order (ascending "line"). Each is a copy of the input line with
              two extra keys:
                "filled"    : int, units allocated
                "backorder" : int, qty - filled
rejections  : list of (line, reason) tuples, reason == "unknown sku" for a
              sku not in stock. Ordered by line number.
"""


def allocate(lines, stock):
    remaining = dict(stock)  # copy: caller's stock dict is never mutated
    ordered = sorted(lines, key=lambda ln: ln["line"])

    allocations = []
    rejections = []
    for ln in ordered:
        sku = ln["sku"]
        if sku not in remaining:
            rejections.append((ln["line"], "unknown sku"))
            continue
        qty = ln["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        out = dict(ln)
        out["filled"] = filled
        out["backorder"] = qty - filled
        allocations.append(out)

    return allocations, rejections
