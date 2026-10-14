"""Parse raw order text into accepted lines and rejections."""


def _is_header(stripped: str) -> bool:
    return "".join(stripped.split()).lower() == "order_id,sku,qty"


def parse(text: str):
    """Return (lines, rejected).

    lines: list of dict with keys n, order, sku, qty (merged, first-appearance order)
    rejected: list of (n, reason) tuples
    """
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    header_checked = False

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not header_checked:
            header_checked = True
            if _is_header(stripped):
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            rejected.append((n, "bad format"))
            continue

        order, sku, qty_text = fields[0], fields[1].upper(), fields[2]
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
