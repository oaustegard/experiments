"""Parse raw order CSV text into merged order lines and rejections."""


def parse(text):
    """parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int],
    rejected:list[tuple n:int reason:str])
    """
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    header_checked = False

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if not header_checked:
            header_checked = True
            if "".join(raw.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3:
            rejected.append((n, "bad format"))
            continue
        order, sku, qty_text = fields
        if order == "" or sku == "":
            rejected.append((n, "bad format"))
            continue

        try:
            qty = int(qty_text)
        except ValueError:
            qty = 0
        if qty <= 0:
            rejected.append((n, "bad qty"))
            continue

        sku = sku.upper()
        order = order.upper()
        key = (order, sku)
        if key in index:
            lines[index[key]]["qty"] += qty
        else:
            index[key] = len(lines)
            lines.append({"n": n, "order": order, "sku": sku, "qty": qty})

    return lines, rejected
