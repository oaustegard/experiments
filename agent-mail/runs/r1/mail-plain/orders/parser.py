"""Parse raw order text into accepted lines and rejections.

Public interface:

    parse(orders_csv: str) -> tuple[list[dict], list[dict]]

lines   : [{"line": int, "order_id": str, "sku": str, "qty": int}, ...]
          merged per (order_id, sku), in first-occurrence order.
rejects : [{"line": int, "reason": "bad format" | "bad qty"}, ...]
"""


def parse(orders_csv: str) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    rejects: list[dict] = []
    index: dict[tuple[str, str], dict] = {}  # (order_id, sku) -> merged line
    header_checked = False

    for lineno, raw in enumerate(orders_csv.split("\n"), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            if "".join(text.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in text.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            rejects.append({"line": lineno, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields[0], fields[1].upper(), fields[2]
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejects.append({"line": lineno, "reason": "bad qty"})
            continue

        key = (order_id, sku)
        if key in index:
            index[key]["qty"] += qty
        else:
            entry = {"line": lineno, "order_id": order_id, "sku": sku, "qty": qty}
            index[key] = entry
            lines.append(entry)

    return lines, rejects
