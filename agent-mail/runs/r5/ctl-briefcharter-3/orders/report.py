"""Entry point: run the order pipeline and render the invoice text."""

from orders.inventory import allocate
from orders.parser import parse
from orders.pricing import price_orders


def process(orders_csv, catalog, stock):
    """Return the invoice text for the given orders, catalog and stock."""
    lines, parse_rejected = parse(orders_csv)
    allocated, stock_rejected = allocate(lines, stock)

    rejected = sorted(list(parse_rejected) + list(stock_rejected), key=lambda r: r[0])

    backorder_by_n = {a["n"]: a["backorder"] for a in allocated}
    priced_input = [
        {"n": a["n"], "order": a["order"], "sku": a["sku"], "filled": a["filled"]}
        for a in allocated
    ]
    priced = price_orders(priced_input, catalog)

    blocks = []
    for order in sorted(priced, key=lambda o: o["order"]):
        out = [f"ORDER {order['order']}"]
        status = "COMPLETE"
        for line in order["lines"]:
            name = catalog[line["sku"]]["name"]
            text = (
                f"  {line['sku']} {name} x{line['filled']} "
                f"@ {line['unit']} = {line['total']}"
            )
            backorder = backorder_by_n[line["n"]]
            if backorder > 0:
                text += f" (backorder {backorder})"
                status = "PARTIAL"
            out.append(text)
        out.append(f"  subtotal {order['subtotal']}")
        out.append(f"  tax {order['tax']}")
        out.append(f"  shipping {order['shipping']}")
        out.append(f"  total {order['total']}")
        out.append(f"  status {status}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
