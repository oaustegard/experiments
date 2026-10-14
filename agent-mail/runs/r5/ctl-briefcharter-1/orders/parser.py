"""Parse raw order text into order lines, collecting rejections."""


def parse(text: str):
    """Parse orders text.

    Returns (lines, rejected):
      lines    -- list of dict with keys n:int, order:str, sku:str, qty:int
      rejected -- list of tuple (n:int, reason:str)
    """
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    seen_content = False  # True once the first non-blank, non-comment line has passed

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or raw.lstrip(" ").startswith("#"):
            continue

        if not seen_content:
            seen_content = True
            if raw.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
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
