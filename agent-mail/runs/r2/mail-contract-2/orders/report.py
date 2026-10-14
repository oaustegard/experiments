"""Entry point: run the order pipeline and render the invoice text.

Pipeline (dave owns the call order):
    parse(text) -> (lines, rejected)
    allocate(lines, stock) -> (allocated, rejected)
    price(allocated, catalog) -> (priced, orders)
"""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price


def process(orders_csv, catalog, stock):
    """process(orders_csv:str, catalog:dict, stock:dict) -> str"""
    lines, rejected = parse(orders_csv)
    allocated, alloc_rejected = allocate(lines, stock)
    rejected = sorted(list(rejected) + list(alloc_rejected), key=lambda r: r[0])

    priced, summaries = price(allocated, catalog)

    backorder_by_n = {a["n"]: a["backorder"] for a in allocated}
    summary_by_order = {s["order"]: s for s in summaries}

    lines_by_order = {}
    for p in priced:
        lines_by_order.setdefault(p["order"], []).append(p)

    blocks = []
    for order_id in sorted(lines_by_order):
        block = [f"ORDER {order_id}"]
        for p in lines_by_order[order_id]:
            name = catalog[p["sku"]]["name"]
            text = (
                f"  {p['sku']} {name} x{p['filled']} @ {p['unit']} = {p['total']}"
            )
            backorder = backorder_by_n.get(p["n"], 0)
            if backorder > 0:
                text += f" (backorder {backorder})"
            block.append(text)
        s = summary_by_order[order_id]
        block.append(f"  subtotal {s['subtotal']}")
        block.append(f"  tax {s['tax']}")
        block.append(f"  shipping {s['shipping']}")
        block.append(f"  total {s['total']}")
        blocks.append("\n".join(block))

    if rejected:
        reject_lines = ["REJECTED"]
        for n, reason in rejected:
            reject_lines.append(f"  line {n}: {reason}")
        blocks.append("\n".join(reject_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
