"""Entry point: run the pipeline and render the invoice text.

Pipeline assumed (interfaces not yet agreed; see render() data shapes):
    parser.parse_orders(orders_csv) -> (order_lines, rejected)
        order_lines: list of dicts {order_id, line_no, sku, qty}
        rejected:    list of (line_no, reason)
    inventory.allocate(order_lines, stock) -> allocated lines with filled/backorder
    pricing.price_order(lines, catalog) -> order dict (see render)
"""
from decimal import Decimal, ROUND_HALF_UP

_CENT = Decimal("0.01")


def _money(value):
    return str(Decimal(value).quantize(_CENT, rounding=ROUND_HALF_UP))


def _order_block(order):
    """order: {"id": str, "lines": [line...], "subtotal", "tax", "shipping", "total"}
    line: {"sku", "name", "filled", "backorder", "unit", "total"}
    Money fields may be Decimal, int or str.
    """
    out = [f"ORDER {order['id']}"]
    for ln in order["lines"]:
        text = (f"  {ln['sku']} {ln['name']} x{ln['filled']} "
                f"@ {_money(ln['unit'])} = {_money(ln['total'])}")
        if ln.get("backorder", 0) > 0:
            text += f" (backorder {ln['backorder']})"
        out.append(text)
    out.append(f"  subtotal {_money(order['subtotal'])}")
    out.append(f"  tax {_money(order['tax'])}")
    out.append(f"  shipping {_money(order['shipping'])}")
    out.append(f"  total {_money(order['total'])}")
    return "\n".join(out)


def render(orders, rejected):
    """Render the invoice text.

    orders:   list of order dicts (see _order_block); orders with no lines are skipped.
    rejected: list of (line_no, reason) tuples.
    """
    blocks = [_order_block(o) for o in sorted(orders, key=lambda o: o["id"]) if o["lines"]]
    if rejected:
        rej = ["REJECTED"]
        for line_no, reason in sorted(rejected, key=lambda r: r[0]):
            rej.append(f"  line {line_no}: {reason}")
        blocks.append("\n".join(rej))
    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)


def process(orders_csv, catalog, stock):
    """Run the pipeline and return the invoice text. Wiring not yet verifiable:
    the parser, inventory and pricing modules were not present when this was written."""
    from orders import parser, inventory, pricing  # noqa: imported lazily

    order_lines, rejected = parser.parse_orders(orders_csv)
    allocated = inventory.allocate(order_lines, stock)
    orders = pricing.price_orders(allocated, catalog)
    return render(orders, rejected)
