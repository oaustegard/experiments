"""Parse raw order CSV text into order lines and rejections.

Interface (owned by alice):

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections).

lines: list of {"line_no": int, "order_id": str, "sku": str, "qty": int}
    Accepted lines, merged on (order_id, sku). A merged line keeps the
    line_no and position of its first occurrence; qty is the sum.
    sku is uppercased.

rejections: list of {"line_no": int, "reason": str, "text": str}
    reason is "bad format" or "bad qty". line_no is physical, 1-based.
"""


def parse_orders(orders_csv):
    lines = []
    rejections = []
    merged = {}  # (order_id, sku) -> index into lines
    header_checked = False

    for line_no, raw in enumerate(orders_csv.split("\n"), start=1):
        text = raw.rstrip("\r")
        stripped = text.strip()
        if stripped == "":
            continue
        if stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            if text.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in text.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejections.append({"line_no": line_no, "reason": "bad format", "text": text})
            continue

        order_id, sku, qty_text = fields[0], fields[1].upper(), fields[2]
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejections.append({"line_no": line_no, "reason": "bad qty", "text": text})
            continue

        key = (order_id, sku)
        if key in merged:
            lines[merged[key]]["qty"] += qty
        else:
            merged[key] = len(lines)
            lines.append({"line_no": line_no, "order_id": order_id, "sku": sku, "qty": qty})

    return lines, rejections
