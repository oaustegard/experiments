"""Allocate stock to accepted order lines.

Contract (called by orders/report.py):

    allocate(lines, stock) -> (allocated, rejections)

lines: list of accepted line dicts, each with keys
    "line_no" (int), "sku" (str), "qty" (int); other keys are carried through.
    Order of the list is not trusted: lines are processed ascending by "line_no".
stock: {sku: int}. Never mutated.

allocated: list of copies of the accepted lines (ascending line_no), each with
    "filled" (int) and "backorder" (int) added. Rejected lines are not included.
rejections: list of {"line_no": int, "reason": "unknown sku"} (ascending line_no).
"""


def allocate(lines, stock):
    remaining = dict(stock)
    allocated = []
    rejections = []
    for line in sorted(lines, key=lambda l: l["line_no"]):
        sku = line["sku"]
        qty = line["qty"]
        if sku not in remaining:
            rejections.append({"line_no": line["line_no"], "reason": "unknown sku"})
            continue
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        out = dict(line)
        out["filled"] = filled
        out["backorder"] = qty - filled
        allocated.append(out)
    return allocated, rejections
