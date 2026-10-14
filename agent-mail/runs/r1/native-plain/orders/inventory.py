"""Stock allocation for accepted order lines.

Contract (agreed with the team):

    allocate(lines, stock) -> (allocated, rejected)

- lines: list of dicts with keys "line_no" (int), "sku" (str), "qty" (int),
  already merged across orders and in any order.
- stock: {sku: int}. Never mutated.
- allocated: list of dicts, one per accepted line, sorted by line_no, each the
  input dict plus "filled" (int) and "backorder" (int).
- rejected: list of (line_no, reason) tuples, sorted by line_no. Reason is
  "unknown sku" for a sku not present in stock. Rejected lines are not in
  allocated.
"""


def allocate(lines, stock):
    remaining = dict(stock)  # copy; the caller's dict is never touched
    allocated = []
    rejected = []

    for line in sorted(lines, key=lambda ln: ln["line_no"]):
        sku = line["sku"]
        if sku not in remaining:
            rejected.append((line["line_no"], "unknown sku"))
            continue
        qty = line["qty"]
        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        result = dict(line)
        result["filled"] = filled
        result["backorder"] = qty - filled
        allocated.append(result)

    return allocated, rejected
