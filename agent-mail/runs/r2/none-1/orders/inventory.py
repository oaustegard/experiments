"""Stock allocation for accepted order lines.

Interface (carol):

    allocate(lines, stock) -> (allocated, rejections)

lines:       accepted lines from parser.parse_orders, each a dict
             {"line": int, "order_id": str, "sku": str, "qty": int}.
stock:       {sku: int}. Never mutated.
allocated:   new dicts, one per line whose sku is in stock, in ascending
             "line" order. Each is a copy of the input line plus
             "filled" (int) and "backorder" (int), where
             filled = min(qty, remaining stock) and backorder = qty - filled.
rejections:  dicts {"line": int, "reason": "unknown sku"}, ascending "line".
"""


def allocate(lines, stock):
    remaining = dict(stock)
    allocated = []
    rejections = []
    for entry in sorted(lines, key=lambda l: l["line"]):
        sku = entry["sku"]
        if sku not in remaining:
            rejections.append({"line": entry["line"], "reason": "unknown sku"})
            continue
        qty = entry["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        out = dict(entry)
        out["filled"] = filled
        out["backorder"] = qty - filled
        allocated.append(out)
    return allocated, rejections
