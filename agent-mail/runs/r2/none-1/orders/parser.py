"""Parse raw order text into order lines and rejections.

Interface (for the other modules):

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections).

lines: accepted order lines, in input order, each a dict:
    {"line": int, "order_id": str, "sku": str, "qty": int}
    - "line" is the 1-based physical line number of the first occurrence.
    - sku is uppercased; order_id is kept as written (stripped).
    - lines with the same (order_id, sku) are merged: qty summed, position
      and line number of the first occurrence kept.

rejections: one dict per rejected line, in input order:
    {"line": int, "reason": str}
    reason is "bad format" or "bad qty".
"""


def parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    rejections: list[dict] = []
    index: dict[tuple[str, str], dict] = {}
    header_pending = True  # only the first non-blank, non-comment line may be a header

    for number, raw in enumerate(orders_csv.split("\n"), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue

        if header_pending:
            header_pending = False
            if raw.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            rejections.append({"line": number, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejections.append({"line": number, "reason": "bad qty"})
            continue

        sku = sku.upper()
        key = (order_id, sku)
        if key in index:
            index[key]["qty"] += qty
        else:
            entry = {"line": number, "order_id": order_id, "sku": sku, "qty": qty}
            index[key] = entry
            lines.append(entry)

    return lines, rejections
