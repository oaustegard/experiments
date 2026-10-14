"""Parse raw order text into accepted lines and rejections."""


def parse_orders(orders_csv):
    """Parse the orders text.

    Returns (lines, rejections):
      lines: list of {"line_no": int, "order_id": str, "sku": str, "qty": int}
             Accepted lines, merged on (order_id, sku). A merged line keeps
             the line number and position of its first occurrence.
      rejections: list of {"line_no": int, "reason": str}, in line order.
             reason is "bad format" or "bad qty".
    """
    lines = []
    rejections = []
    merged = {}
    header_allowed = True

    for number, raw in enumerate(orders_csv.splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue

        if header_allowed:
            header_allowed = False
            if text.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [field.strip() for field in text.split(",")]
        if len(fields) != 3:
            rejections.append({"line_no": number, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields
        if not order_id or not sku:
            rejections.append({"line_no": number, "reason": "bad format"})
            continue

        try:
            qty = int(qty_text)
        except ValueError:
            qty = None
        if qty is None or qty <= 0:
            rejections.append({"line_no": number, "reason": "bad qty"})
            continue

        sku = sku.upper()
        key = (order_id, sku)
        if key in merged:
            merged[key]["qty"] += qty
        else:
            entry = {"line_no": number, "order_id": order_id, "sku": sku, "qty": qty}
            merged[key] = entry
            lines.append(entry)

    return lines, rejections
