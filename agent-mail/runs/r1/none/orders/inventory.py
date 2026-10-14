"""Stock allocation for accepted order lines.

Interface (agreed shape, documented for the other modules):

    allocate(lines, stock) -> (allocations, rejections)

lines: iterable of accepted order lines. Each line is a dict or an object with
    attributes: ``line_no`` (int, line number in the source file), ``sku`` (str),
    ``qty`` (int). Lines may come from several orders; they are processed in
    ascending ``line_no`` order.
stock: {sku: int} units on hand. Never mutated.

Returns:
    allocations: list of dicts, one per allocated line, in processing order:
        {"line_no": int, "sku": str, "qty": int, "filled": int, "backorder": int}
    rejections: list of dicts, one per rejected line:
        {"line_no": int, "sku": str, "reason": "unknown sku"}
"""


def _field(line, name):
    if isinstance(line, dict):
        return line[name]
    return getattr(line, name)


def allocate(lines, stock):
    remaining = dict(stock)  # copy: the caller's dict must not change
    allocations = []
    rejections = []

    for line in sorted(lines, key=lambda ln: _field(ln, "line_no")):
        line_no = _field(line, "line_no")
        sku = _field(line, "sku")
        qty = _field(line, "qty")

        if sku not in remaining:
            rejections.append({"line_no": line_no, "sku": sku, "reason": "unknown sku"})
            continue

        filled = min(qty, remaining[sku])
        remaining[sku] -= filled
        allocation = {
            "line_no": line_no,
            "sku": sku,
            "qty": qty,
            "filled": filled,
            "backorder": qty - filled,
        }
        if isinstance(line, dict) and "order_id" in line:
            allocation["order_id"] = line["order_id"]
        allocations.append(allocation)

    return allocations, rejections
