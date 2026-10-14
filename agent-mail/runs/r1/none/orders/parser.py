"""Parse raw order text into order lines and rejections.

Public interface:

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections).

lines: list of accepted, merged order lines in order of first appearance:
    {"line_no": int, "order_id": str, "sku": str, "qty": int}
    line_no is the physical 1-based line number of the first accepted line
    for that (order_id, sku); qty is the sum over all merged lines.

rejections: list of rejected data lines in file order:
    {"line_no": int, "reason": str}
    reason is "bad format" or "bad qty".
"""

HEADER = "order_id,sku,qty"


def parse_orders(orders_csv: str):
    lines = []
    rejections = []
    index_by_key = {}  # (order_id, sku) -> position in `lines`
    header_pending = True  # the header may only be the first non-blank, non-comment line

    for line_no, raw in enumerate(orders_csv.splitlines(), start=1):
        if raw.strip() == "":
            continue
        if raw.lstrip(" ").startswith("#"):
            continue

        if header_pending:
            header_pending = False
            if raw.replace(" ", "").lower() == HEADER:
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejections.append({"line_no": line_no, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields[0], fields[1].upper(), fields[2]
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejections.append({"line_no": line_no, "reason": "bad qty"})
            continue

        key = (order_id, sku)
        if key in index_by_key:
            lines[index_by_key[key]]["qty"] += qty
        else:
            index_by_key[key] = len(lines)
            lines.append({"line_no": line_no, "order_id": order_id,
                          "sku": sku, "qty": qty})

    return lines, rejections
