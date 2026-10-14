"""Parse raw order text into merged order lines and rejections.

Contract:
    parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int],
                        rejected:list[tuple n:int reason:str])
"""


def parse(text):
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    header_pending = True  # the first content line may be a header

    for n, raw in enumerate(text.split("\n"), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if header_pending:
            header_pending = False
            if raw.replace(" ", "").lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3:
            rejected.append((n, "bad format"))
            continue
        order, sku, qty_text = fields
        if not order or not sku:
            rejected.append((n, "bad format"))
            continue
        try:
            qty = int(qty_text)
        except ValueError:
            rejected.append((n, "bad qty"))
            continue
        if qty <= 0:
            rejected.append((n, "bad qty"))
            continue

        sku = sku.upper()
        key = (order, sku)
        if key in index:
            lines[index[key]]["qty"] += qty
        else:
            index[key] = len(lines)
            lines.append({"n": n, "order": order, "sku": sku, "qty": qty})

    return lines, rejected
