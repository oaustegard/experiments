"""Parse raw order text into merged order lines and rejections."""

HEADER = "order_id,sku,qty"


def parse(text: str):
    """Parse order CSV text.

    Returns (lines, rejected):
      lines: list of dict with keys n:int, order:str, sku:str, qty:int
      rejected: list of (n:int, reason:str)
    """
    lines = []
    rejected = []
    merged = {}  # (order, sku) -> line dict; insertion order = first occurrence
    header_checked = False

    for n, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            if raw.replace(" ", "").lower() == HEADER:
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
        if key in merged:
            merged[key]["qty"] += qty
        else:
            line = {"n": n, "order": order, "sku": sku, "qty": qty}
            merged[key] = line
            lines.append(line)

    return lines, rejected
