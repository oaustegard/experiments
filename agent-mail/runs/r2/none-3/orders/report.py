"""Entry point: run the order pipeline and render the invoice text (owner: dave).

Pipeline (interfaces as written by the owners):

    parser.parse_orders(orders_csv)       -> (lines, rejections)
    inventory.allocate(lines, stock)      -> (allocated, rejections)
    pricing.price_order(lines, catalog)   -> {"lines", "subtotal", "tax", "shipping", "total"}
"""

from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse_orders
from orders.inventory import allocate
from orders.pricing import price_order

CENT = Decimal("0.01")


def _money(value):
    return str(Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP))


def process(orders_csv, catalog, stock):
    lines, rejections = parse_orders(orders_csv)
    allocated, alloc_rejections = allocate(lines, stock)

    rejected = [(r["line_no"], r["reason"]) for r in rejections]
    rejected += [(r["line_no"], r["reason"]) for r in alloc_rejections]

    by_order = {}
    for line in allocated:
        by_order.setdefault(line["order_id"], []).append(line)

    blocks = []
    for order_id in sorted(by_order):
        order_lines = by_order[order_id]
        priced = price_order(order_lines, catalog)
        out = [f"ORDER {order_id}"]
        for line, p in zip(order_lines, priced["lines"]):
            name = catalog[line["sku"]]["name"]
            text = (
                f"  {line['sku']} {name} x{line['filled']} "
                f"@ {_money(p['unit_price'])} = {_money(p['line_total'])}"
            )
            if line["backorder"] > 0:
                text += f" (backorder {line['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(priced['subtotal'])}")
        out.append(f"  tax {_money(priced['tax'])}")
        out.append(f"  shipping {_money(priced['shipping'])}")
        out.append(f"  total {_money(priced['total'])}")
        blocks.append("\n".join(out))

    if rejected:
        rej = ["REJECTED"]
        for line_no, reason in sorted(rejected, key=lambda r: r[0]):
            rej.append(f"  line {line_no}: {reason}")
        blocks.append("\n".join(rej))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
