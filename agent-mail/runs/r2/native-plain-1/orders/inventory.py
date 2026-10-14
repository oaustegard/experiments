"""Allocate stock to accepted order lines.

Interface (proposed to the team):

    allocate(lines: list[dict], stock: dict[str, int]) -> tuple[list[dict], list[dict]]

`lines` is the accepted (already merged) order lines in input order. Each dict
has at least:
    "line_no": int   line number in the input file
    "sku": str
    "qty": int
Any other keys (order id, etc.) are carried through untouched.

Returns (allocations, rejections):
    allocations: one dict per accepted line, the input dict plus
                 "filled": int and "backorder": int
    rejections:  one dict per rejected line: {"line_no": int, "reason": str}

The caller's `stock` dict is never mutated.
"""


def allocate(lines, stock):
    remaining = dict(stock)
    ordered = sorted(lines, key=lambda line: line["line_no"])
    allocations = []
    rejections = []
    for line in ordered:
        sku = line["sku"]
        if sku not in remaining:
            rejections.append({"line_no": line["line_no"], "reason": "unknown sku"})
            continue
        qty = line["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        allocated = dict(line)
        allocated["filled"] = filled
        allocated["backorder"] = qty - filled
        allocations.append(allocated)
    return allocations, rejections
