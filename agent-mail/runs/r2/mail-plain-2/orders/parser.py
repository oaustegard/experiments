"""Parse raw order text into order lines and rejections.

Public interface:

    parse_lines(orders_csv: str) -> tuple[list[dict], list[dict]]

Returns (lines, rejections):

- lines: accepted order lines, merged on (order_id, sku), in order of first
  occurrence. Each is {"line": int, "order": str, "sku": str, "qty": int}
  where "line" is the physical 1-based line number of the first occurrence.
- rejections: {"line": int, "reason": str} for each rejected data line, in
  line order. Reason is "bad format" or "bad qty".
"""


HEADER = "order_id,sku,qty"


def parse_lines(orders_csv: str) -> tuple[list[dict], list[dict]]:
    lines: list[dict] = []
    rejections: list[dict] = []
    merged: dict[tuple[str, str], dict] = {}
    header_pending = True  # the first non-blank, non-comment line may be a header

    for number, raw in enumerate(orders_csv.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "" or raw.lstrip().startswith("#"):
            continue

        if header_pending:
            header_pending = False
            if raw.replace(" ", "").lower() == HEADER:
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejections.append({"line": number, "reason": "bad format"})
            continue

        order_id, sku, qty_text = fields
        sku = sku.upper()
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejections.append({"line": number, "reason": "bad qty"})
            continue

        key = (order_id, sku)
        if key in merged:
            merged[key]["qty"] += qty
        else:
            entry = {"line": number, "order": order_id, "sku": sku, "qty": qty}
            merged[key] = entry
            lines.append(entry)

    return lines, rejections
