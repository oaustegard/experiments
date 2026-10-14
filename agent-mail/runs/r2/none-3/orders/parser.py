"""Parse raw order text into lines and rejections (owner: alice).

Interface (for the other modules):

    parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections):

    lines      = [{"line_no": int, "order_id": str, "sku": str, "qty": int}, ...]
                 accepted lines, merged on (order_id, sku), in input order;
                 a merged line keeps the line_no and position of its first line.
    rejections = [{"line_no": int, "reason": str}, ...]
                 reason is "bad format" or "bad qty"; in input order.

Line numbers are physical, 1-based, and count every line.
"""


HEADER = "order_id,sku,qty"


def parse_orders(orders_csv: str) -> tuple[list[dict], list[dict]]:
    lines = []
    rejections = []
    merged = {}  # (order_id, sku) -> index into lines
    header_pending = True  # only the first non-blank, non-comment line may be a header

    for line_no, raw in enumerate(orders_csv.split("\n"), start=1):
        raw = raw.rstrip("\r")
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
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
        if key in merged:
            lines[merged[key]]["qty"] += qty
        else:
            merged[key] = len(lines)
            lines.append({"line_no": line_no, "order_id": order_id, "sku": sku, "qty": qty})

    return lines, rejections
