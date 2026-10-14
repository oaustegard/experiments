"""Entry point: run the order pipeline and render the invoice text.

process(orders_csv, catalog, stock) -> str

Pipeline:
    parser.parse(text) -> (lines, rejected)
    inventory.allocate(lines, stock) -> (allocs, rejected)
    pricing.price_line(sku, filled, catalog) -> {'unit_price', 'line_total'}
    pricing.price_order(line_totals) -> {'subtotal', 'tax', 'shipping', 'total'}
"""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_line, price_order


def _money(value):
    return "{:.2f}".format(value)


def process(orders_csv, catalog, stock):
    lines, parse_rejected = parse(orders_csv)
    allocs, alloc_rejected = allocate(lines, stock)

    rejected = sorted(list(parse_rejected) + list(alloc_rejected),
                      key=lambda r: r[0])

    # Group allocated (accepted) lines by order id.
    by_order = {}
    for a in allocs:
        by_order.setdefault(a["order"], []).append(a)

    blocks = []
    for order_id in sorted(by_order):
        order_lines = by_order[order_id]
        out = ["ORDER {}".format(order_id)]
        line_totals = []
        partial = False
        for a in order_lines:
            sku = a["sku"]
            filled = a["filled"]
            backorder = a["backorder"]
            priced = price_line(sku, filled, catalog)
            line_totals.append(priced["line_total"])
            name = catalog.get(sku, {}).get("name", sku)
            text = "  {} {} x{} @ {} = {}".format(
                sku, name, filled,
                _money(priced["unit_price"]),
                _money(priced["line_total"]),
            )
            if backorder > 0:
                text += " (backorder {})".format(backorder)
                partial = True
            out.append(text)

        totals = price_order(line_totals)
        out.append("  subtotal {}".format(_money(totals["subtotal"])))
        out.append("  tax {}".format(_money(totals["tax"])))
        out.append("  shipping {}".format(_money(totals["shipping"])))
        out.append("  total {}".format(_money(totals["total"])))
        out.append("  status {}".format("PARTIAL" if partial else "COMPLETE"))
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append("  line {}: {}".format(n, reason))
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"

    return "\n\n".join(blocks)
