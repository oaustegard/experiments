"""Parse raw order text into order lines and rejections.

Interface (owned by alice):

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections):
  lines      = [{"line": int, "order": str, "sku": str, "qty": int}, ...]
               accepted lines, merged on (order_id, sku), in first-seen order.
  rejections = [{"line": int, "reason": str}, ...]
               reason is "bad format" or "bad qty". Line numbers are physical, 1-based.
"""


def parse_orders(orders_csv):
    lines = []
    rejections = []
    index_by_key = {}
    header_checked = False

    for lineno, raw in enumerate(orders_csv.split("\n"), start=1):
        text = raw.rstrip("\r")

        # Blank lines and comment lines are ignored.
        if text.strip() == "" or text.lstrip().startswith("#"):
            continue

        # The first non-blank, non-comment line may be a header.
        if not header_checked:
            header_checked = True
            if "".join(text.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in text.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            rejections.append({"line": lineno, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields[0], fields[1].upper(), fields[2]
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejections.append({"line": lineno, "reason": "bad qty"})
            continue

        key = (order_id, sku)
        if key in index_by_key:
            lines[index_by_key[key]]["qty"] += qty
        else:
            index_by_key[key] = len(lines)
            lines.append({"line": lineno, "order": order_id, "sku": sku, "qty": qty})

    return lines, rejections
