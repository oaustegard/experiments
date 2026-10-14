"""Entry point: run the order pipeline and render the invoice text."""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_orders


def _line_text(line, catalog, backorder):
    name = catalog[line["sku"]]["name"]
    text = f'  {line["sku"]} {name} x{line["filled"]} @ {line["unit"]} = {line["total"]}'
    bo = backorder.get(line["n"], 0)
    if bo > 0:
        text += f" (backorder {bo})"
    return text


def _order_block(order_id, priced, catalog, backorder):
    info = priced[order_id]
    rows = [f"ORDER {order_id}"]
    for line in sorted(info["lines"], key=lambda x: x["n"]):
        rows.append(_line_text(line, catalog, backorder))
    rows.append(f'  subtotal {info["subtotal"]}')
    rows.append(f'  tax {info["tax"]}')
    rows.append(f'  shipping {info["shipping"]}')
    rows.append(f'  total {info["total"]}')
    return "\n".join(rows)


def process(orders_csv, catalog, stock):
    lines, rejected = parse(orders_csv)

    rejected = list(rejected)
    known = []
    for line in lines:
        if line["sku"] in catalog:
            known.append(line)
        else:
            rejected.append((line["n"], "unknown sku"))

    allocs, alloc_rejected = allocate(known, stock)
    rejected.extend(alloc_rejected)

    backorder = {a["n"]: a["backorder"] for a in allocs}
    priced = price_orders(allocs, catalog)

    blocks = [
        _order_block(order_id, priced, catalog, backorder)
        for order_id in sorted(priced)
    ]

    if rejected:
        rows = ["REJECTED"]
        for n, reason in sorted(rejected, key=lambda r: r[0]):
            rows.append(f"  line {n}: {reason}")
        blocks.append("\n".join(rows))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
