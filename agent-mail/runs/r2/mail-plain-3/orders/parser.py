"""Parse raw order CSV text into accepted lines and rejections.

API:
    parse(orders_csv: str) -> (accepted, rejected)

accepted: list of {"order": str, "line": int, "sku": str, "qty": int},
    in file order, with duplicate (order, sku) pairs merged.
rejected: list of {"line": int, "reason": "bad format" | "bad qty"},
    in file order.
"""

HEADER = "order_id,sku,qty"


def _is_skipped(text):
    """Blank lines and comment lines are ignored."""
    stripped = text.strip()
    return stripped == "" or stripped.startswith("#")


def parse(orders_csv):
    accepted = []
    rejected = []
    merged = {}  # (order, sku) -> entry in accepted
    header_checked = False

    for line_no, raw in enumerate(orders_csv.splitlines(), start=1):
        if _is_skipped(raw):
            continue

        if not header_checked:
            header_checked = True
            if "".join(raw.split()).lower() == HEADER:
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3:
            rejected.append({"line": line_no, "reason": "bad format"})
            continue

        order, sku, qty_text = fields
        if not order or not sku:
            rejected.append({"line": line_no, "reason": "bad format"})
            continue

        try:
            qty = int(qty_text)
        except ValueError:
            rejected.append({"line": line_no, "reason": "bad qty"})
            continue
        if qty <= 0:
            rejected.append({"line": line_no, "reason": "bad qty"})
            continue

        sku = sku.upper()
        key = (order, sku)
        if key in merged:
            merged[key]["qty"] += qty
        else:
            entry = {"order": order, "line": line_no, "sku": sku, "qty": qty}
            merged[key] = entry
            accepted.append(entry)

    return accepted, rejected
