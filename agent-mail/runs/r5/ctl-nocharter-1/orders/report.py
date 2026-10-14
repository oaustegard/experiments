"""Entry point: run the order pipeline and render the invoice text."""

from decimal import Decimal

from orders.inventory import allocate
from orders.parser import parse
from orders.pricing import price_orders


def _money(value) -> str:
    return f"{Decimal(value):.2f}"


def process(orders_csv: str, catalog: dict, stock: dict) -> str:
    lines, rejected = parse(orders_csv)
    allocated, allocate_rejected = allocate(lines, stock)
    rejected = rejected + allocate_rejected

    # Group allocated lines by order, keeping input order within each order.
    grouped = {}
    for line in allocated:
        grouped.setdefault(line["order"], []).append(line)

    priced = price_orders(grouped, catalog)

    blocks = []
    for order in sorted(priced, key=lambda o: o["order"]):
        source = grouped[order["order"]]
        out = [f"ORDER {order['order']}"]
        for line, src in zip(order["lines"], source):
            text = (
                f"  {line['sku']} {line['name']} x{line['filled']}"
                f" @ {_money(line['unit'])} = {_money(line['total'])}"
            )
            if src["backorder"] > 0:
                text += f" (backorder {src['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(order['subtotal'])}")
        out.append(f"  tax {_money(order['tax'])}")
        out.append(f"  shipping {_money(order['shipping'])}")
        out.append(f"  total {_money(order['total'])}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in sorted(rejected):
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
