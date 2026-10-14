"""Parse raw order text into order lines and rejections.

Contract:
    parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int],
                        rejected:list[tuple n:int reason:str])
"""


def parse(text):
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    header_checked = False

    for n, raw in enumerate(text.split("\n"), start=1):
        raw = raw.rstrip("\r")
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        if not header_checked:
            header_checked = True
            if "".join(stripped.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in stripped.split(",")]
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
        if key in index:
            lines[index[key]]["qty"] += qty
        else:
            index[key] = len(lines)
            lines.append({"n": n, "order": order, "sku": sku, "qty": qty})

    return lines, rejected
