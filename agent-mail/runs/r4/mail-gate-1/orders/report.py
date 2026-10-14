"""Entry point: run the order pipeline and render the invoice text."""
from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_line, price_order


def _money(value):
    """Format a decimal string or number with exactly 2 decimals."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def process(orders_csv, catalog, stock):
    lines, parse_rejected = parse(orders_csv)
    allocs, alloc_rejected = allocate(lines, stock)

    rejected = sorted(list(parse_rejected) + list(alloc_rejected), key=lambda r: r[0])

    # Group allocated lines by order, keeping input order within each order.
    by_order = {}
    for a in allocs:
        by_order.setdefault(a["order"], []).append(a)

    blocks = []
    for order_id in sorted(by_order):
        order_lines = by_order[order_id]
        priced = price_order(
            [{"sku": a["sku"], "filled": a["filled"]} for a in order_lines],
            catalog,
        )
        out = ["ORDER %s" % order_id]
        for a in order_lines:
            sku = a["sku"]
            name = catalog[sku]["name"]
            # Unit price as the pricing module sees it for one unit (0.00 for free SKUs).
            unit = _money(price_line(sku, 1, catalog))
            total = _money(price_line(sku, a["filled"], catalog))
            text = "  %s %s x%d @ %s = %s" % (sku, name, a["filled"], unit, total)
            if a["backorder"] > 0:
                text += " (backorder %d)" % a["backorder"]
            out.append(text)
        out.append("  subtotal %s" % _money(priced["subtotal"]))
        out.append("  tax %s" % _money(priced["tax"]))
        out.append("  shipping %s" % _money(priced["shipping"]))
        out.append("  total %s" % _money(priced["total"]))
        has_backorder = any(a["backorder"] > 0 for a in order_lines)
        out.append("  status %s" % ("PARTIAL" if has_backorder else "COMPLETE"))
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append("  line %d: %s" % (n, reason))
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
