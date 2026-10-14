"""Parse raw order text into merged order lines and rejections.

Contract:
    parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int],
                         rejected:list[tuple n:int reason:str])
"""


def parse(text):
    lines = []
    rejected = []
    merged = {}  # (order, sku) -> index into lines
    header_pending = True  # the first real line may be a header

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if header_pending:
            header_pending = False
            if "".join(raw.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
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
            lines[merged[key]]["qty"] += qty
        else:
            merged[key] = len(lines)
            lines.append({"n": n, "order": order, "sku": sku, "qty": qty})

    return lines, rejected
