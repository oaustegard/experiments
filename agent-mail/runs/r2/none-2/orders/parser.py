"""Parse the raw orders CSV text into order lines and rejections.

Public interface (used by orders/report.py):

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections).

lines: list of dicts, in order of first appearance, one per (order_id, sku):
    {"line_no": int, "order_id": str, "sku": str, "qty": int}
    - line_no is the 1-based physical line number of the first accepted line
      for that (order_id, sku); qty is the sum over all merged lines.
    - sku is uppercased; order_id is kept as written (stripped).

rejections: list of dicts, in line order:
    {"line_no": int, "reason": str}
    reason is "bad format" or "bad qty".
"""


HEADER = "order_id,sku,qty"


def parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    rejections: list[dict] = []
    merged: dict[tuple[str, str], dict] = {}
    header_checked = False

    for line_no, raw in enumerate(orders_csv.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
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
            rejections.append({"line_no": line_no, "reason": "bad qty"})
            continue
        if qty <= 0:
            rejections.append({"line_no": line_no, "reason": "bad qty"})
            continue

        key = (order_id, sku)
        if key in merged:
            merged[key]["qty"] += qty
        else:
            entry = {"line_no": line_no, "order_id": order_id, "sku": sku, "qty": qty}
            merged[key] = entry
            lines.append(entry)

    return lines, rejections
