"""Parse raw order CSV text into merged order lines and rejections."""


def parse(text):
    """parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int],
    rejected:list[tuple n:int reason:str])

    Line numbers are physical, 1-based. Blank and comment lines are ignored.
    The first remaining line is skipped if it is the header order_id,sku,qty.
    Lines with the same (order, sku) are merged; the merged line keeps the
    line number and position of the first occurrence.
    """
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    seen_content = False

    for n, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.lstrip().startswith("#"):
            continue

        if not seen_content:
            seen_content = True
            if raw.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            rejected.append((n, "bad format"))
            continue

        order, sku, qty_text = fields
        sku = sku.upper()
        try:
            qty = int(qty_text)
        except ValueError:
            rejected.append((n, "bad qty"))
            continue
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
