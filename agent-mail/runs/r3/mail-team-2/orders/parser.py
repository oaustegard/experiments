"""Parse raw order CSV text into merged order lines and rejections."""


def parse(text):
    """Return (lines, rejected).

    lines: list of dict with keys n, order, sku, qty (file order, merged).
    rejected: list of (n, reason) tuples in file order.
    """
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    header_checked = False

    for n, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        if not header_checked:
            header_checked = True
            if "".join(stripped.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejected.append((n, "bad format"))
            continue

        order, sku, qty_text = fields[0].upper(), fields[1].upper(), fields[2]
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejected.append((n, "bad qty"))
            continue

        key = (order, sku)
        if key in index:
            lines[index[key]]["qty"] += qty
        else:
            index[key] = len(lines)
            lines.append({"n": n, "order": order, "sku": sku, "qty": qty})

    return lines, rejected
