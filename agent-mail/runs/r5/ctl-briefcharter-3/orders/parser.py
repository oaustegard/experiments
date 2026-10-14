"""Parse raw order CSV text into merged order lines and rejections."""


def parse(text: str):
    lines = []
    rejected = []
    index = {}  # (order, sku) -> position in lines
    seen_content = False  # becomes True after the first blank/comment-free line

    for n, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped == "" or stripped.startswith("#"):
            continue

        if not seen_content:
            seen_content = True
            if "".join(stripped.split()).lower() == "order_id,sku,qty":
                continue

        fields = [f.strip() for f in raw.split(",")]
        if len(fields) != 3 or fields[0] == "" or fields[1] == "":
            rejected.append((n, "bad format"))
            continue

        order = fields[0].upper()
        sku = fields[1].upper()
        try:
            qty = int(fields[2])
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
