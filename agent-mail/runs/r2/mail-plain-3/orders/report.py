"""Entry point: run the order pipeline and render the invoice text.

    process(orders_csv, catalog, stock) -> str

Pipeline: parser.parse -> inventory.allocate -> pricing.price_order -> render.
"""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_order


def _money(value):
    return f"{value:.2f}"


def _order_block(order_id, lines, catalog):
    priced = price_order(lines, catalog)
    out = [f"ORDER {order_id}"]
    for src, p in zip(lines, priced["lines"]):
        name = catalog[p["sku"]]["name"]
        row = (
            f"  {p['sku']} {name} x{p['filled']} "
            f"@ {_money(p['unit_price'])} = {_money(p['line_total'])}"
        )
        backorder = int(src.get("backorder", 0))
        if backorder > 0:
            row += f" (backorder {backorder})"
        out.append(row)
    out.append(f"  subtotal {_money(priced['subtotal'])}")
    out.append(f"  tax {_money(priced['tax'])}")
    out.append(f"  shipping {_money(priced['shipping'])}")
    out.append(f"  total {_money(priced['total'])}")
    return "\n".join(out)


def process(orders_csv, catalog, stock):
    accepted, parser_rejected = parse(orders_csv)
    allocated, alloc_rejected = allocate(accepted, stock)

    # Group allocated lines by order, keeping input order within an order.
    by_order = {}
    for line in allocated:
        by_order.setdefault(line["order"], []).append(line)

    blocks = [
        _order_block(order_id, by_order[order_id], catalog)
        for order_id in sorted(by_order)
    ]

    rejected = sorted(
        list(parser_rejected) + list(alloc_rejected),
        key=lambda r: r["line"],
    )
    if rejected:
        rej_lines = ["REJECTED"]
        for r in rejected:
            rej_lines.append(f"  line {r['line']}: {r['reason']}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
