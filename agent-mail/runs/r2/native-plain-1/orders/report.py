"""Entry point: run the order pipeline and render the invoice text (owned by dave).

    process(orders_csv: str, catalog: dict, stock: dict) -> str

Pipeline:
    parser.parse_orders(orders_csv)      -> (lines, parse_rejections)
    unknown-to-catalog lines             -> rejected "unknown sku"
    inventory.allocate(lines, stock)     -> (allocations, stock_rejections)
    pricing.price_order(catalog, pairs)  -> totals per order
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

    accepted = []
    for line in lines:
        if line["sku"] in catalog:
            accepted.append(line)
        else:
            rejections.append({"line_no": line["line_no"], "reason": "unknown sku"})

    allocations, stock_rejections = allocate(accepted, stock)
    rejections.extend(
        {"line_no": r["line_no"], "reason": r["reason"]} for r in stock_rejections
    )

    orders = {}
    for alloc in allocations:
        orders.setdefault(alloc["order_id"], []).append(alloc)

    blocks = []
    for order_id in sorted(orders):
        order_lines = sorted(orders[order_id], key=lambda a: a["line_no"])
        priced = price_order(
            catalog, [(a["sku"], a["filled"]) for a in order_lines]
        )
        out = [f"ORDER {order_id}"]
        for alloc, line_total in zip(order_lines, priced["lines"]):
            item = catalog[alloc["sku"]]
            text = (
                f"  {alloc['sku']} {item['name']} x{alloc['filled']} "
                f"@ {_money(item['price'])} = {_money(line_total)}"
            )
            if alloc["backorder"] > 0:
                text += f" (backorder {alloc['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(priced['subtotal'])}")
        out.append(f"  tax {_money(priced['tax'])}")
        out.append(f"  shipping {_money(priced['shipping'])}")
        out.append(f"  total {_money(priced['total'])}")
        blocks.append("\n".join(out))

    if rejections:
        rej_lines = ["REJECTED"]
        for r in sorted(rejections, key=lambda r: r["line_no"]):
            rej_lines.append(f"  line {r['line_no']}: {r['reason']}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
