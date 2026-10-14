"""Entry point: run the order pipeline and render the invoice text.

Pipeline (owned by the team, see each module):
    parser.parse_orders(orders_csv)            -> lines, rejections
    inventory.allocate(lines, stock)           -> allocated, rejections
    pricing.price(allocated, catalog)          -> priced_lines, totals

Public interface (owned by dave):

    process(orders_csv: str, catalog: dict, stock: dict) -> str
"""

from decimal import Decimal

from orders.inventory import allocate
from orders.parser import parse_orders
from orders.pricing import price


def _money(value):
    return f"{Decimal(value):.2f}"


def _order_block(order_id, rows, catalog):
    priced_lines, totals = price(rows, catalog)
    priced = totals[order_id]
    out = [f"ORDER {order_id}"]
    for row, priced_line in zip(rows, priced_lines):
        name = catalog[row["sku"]]["name"]
        text = (
            f"  {row['sku']} {name} x{row['filled']}"
            f" @ {_money(priced_line['unit_price'])}"
            f" = {_money(priced_line['line_total'])}"
        )
        if row["backorder"] > 0:
            text += f" (backorder {row['backorder']})"
        out.append(text)
    out.append(f"  subtotal {_money(priced['subtotal'])}")
    out.append(f"  tax {_money(priced['tax'])}")
    out.append(f"  shipping {_money(priced['shipping'])}")
    out.append(f"  total {_money(priced['total'])}")
    return "\n".join(out)


def _rejected_block(rejections):
    out = ["REJECTED"]
    for rej in sorted(rejections, key=lambda r: r["line"]):
        out.append(f"  line {rej['line']}: {rej['reason']}")
    return "\n".join(out)


def process(orders_csv, catalog, stock):
    lines, rejections = parse_orders(orders_csv)

    # A sku must be in both stock (allocate rejects it) and catalog (pricing
    # needs its price and name). Either missing is 'unknown sku'.
    known = []
    for line in lines:
        if line["sku"] in catalog:
            known.append(line)
        else:
            rejections.append({"line": line["line"], "reason": "unknown sku"})

    allocated, alloc_rejections = allocate(known, stock)
    rejections.extend(alloc_rejections)

    by_order = {}
    for row in allocated:
        by_order.setdefault(row["order"], []).append(row)

    blocks = [
        _order_block(order_id, by_order[order_id], catalog)
        for order_id in sorted(by_order)
    ]
    if rejections:
        blocks.append(_rejected_block(rejections))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
