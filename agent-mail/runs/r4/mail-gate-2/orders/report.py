"""Entry point: run the order pipeline and render the invoice text.

Contract:
    process(orders_csv: str, catalog: dict, stock: dict) -> str

Pipeline: parser.parse -> inventory.allocate -> pricing.price -> pricing.order_totals -> render.
"""
from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price, order_totals


def process(orders_csv, catalog, stock):
    lines, parse_rejected = parse(orders_csv)
    allocs, alloc_rejected = allocate(lines, stock)
    priced = price(allocs, catalog)
    totals = order_totals(priced)

    rejected = sorted(list(parse_rejected) + list(alloc_rejected), key=lambda r: r[0])

    # group priced lines by order, keeping input (line number) order within an order
    by_order = {}
    for line in sorted(priced, key=lambda l: l["n"]):
        by_order.setdefault(line["order"], []).append(line)

    blocks = []
    for order in sorted(by_order):
        order_lines = by_order[order]
        if not order_lines:
            continue
        out = ["ORDER " + order]
        for line in order_lines:
            text = "  %s %s x%d @ %s = %s" % (
                line["sku"], line["name"], line["filled"], line["unit"], line["total"])
            if line["backorder"] > 0:
                text += " (backorder %d)" % line["backorder"]
            out.append(text)
        t = totals[order]
        out.append("  subtotal " + t["subtotal"])
        out.append("  tax " + t["tax"])
        out.append("  shipping " + t["shipping"])
        out.append("  total " + t["total"])
        partial = any(l["backorder"] > 0 for l in order_lines)
        out.append("  status PARTIAL" if partial else "  status COMPLETE")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append("  line %d: %s" % (n, reason))
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
