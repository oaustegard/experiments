"""Entry point: run the order pipeline and render the invoice.

Pipeline: parse -> allocate -> price -> order_totals -> render.

    process(orders_csv:str, catalog:dict, stock:dict) -> str
"""
from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price, order_totals


def _name(sku, catalog):
    entry = catalog.get(sku)
    return entry["name"] if entry else sku


def process(orders_csv, catalog, stock):
    lines, parse_rejected = parse(orders_csv)
    allocated, alloc_rejected = allocate(lines, stock)
    rejected = sorted(list(parse_rejected) + list(alloc_rejected), key=lambda r: r[0])

    priced = price(allocated, catalog)  # keeps input order
    totals = order_totals(allocated, catalog)
    backorder_by_n = {ln["n"]: ln["backorder"] for ln in allocated}

    by_order = {}
    for line in priced:
        by_order.setdefault(line["order"], []).append(line)

    blocks = []
    for order_id in sorted(by_order):
        out = [f"ORDER {order_id}"]
        partial = False
        for line in by_order[order_id]:
            text = (
                f"  {line['sku']} {_name(line['sku'], catalog)} "
                f"x{line['filled']} @ {line['unit']} = {line['total']}"
            )
            backorder = backorder_by_n.get(line["n"], 0)
            if backorder > 0:
                partial = True
                text += f" (backorder {backorder})"
            out.append(text)
        t = totals[order_id]
        out.append(f"  subtotal {t['subtotal']}")
        out.append(f"  tax {t['tax']}")
        out.append(f"  shipping {t['shipping']}")
        out.append(f"  total {t['total']}")
        out.append("  status PARTIAL" if partial else "  status COMPLETE")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
