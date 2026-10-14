"""Parse raw order text into merged order lines and rejections."""


def parse(text: str):
    """Return (lines, rejected).

    lines: list of dict with keys n, order, sku, qty (merged by order+sku).
    rejected: list of (n, reason) tuples.
    """
    lines = []
    rejected = []
    index = {}
    header_checked = False

    for n, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            if stripped.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in stripped.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejected.append((n, "bad format"))
            continue

        order, sku, qty_text = fields
        order = order.upper()
        sku = sku.upper()
        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejected.append((n, "bad qty"))
            continue

        key = (order, sku)
        if key in index:
            index[key]["qty"] += qty
        else:
            entry = {"n": n, "order": order, "sku": sku, "qty": qty}
            index[key] = entry
            lines.append(entry)

    return lines, rejected
