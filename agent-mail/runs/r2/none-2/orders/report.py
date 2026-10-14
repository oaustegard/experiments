"""Entry point: run the order pipeline and render the invoice text.

Assumed teammate interfaces (not yet confirmed; no teammate file existed when
this was written):
  parser.parse(orders_csv) -> (lines, rejected)
      lines: list of dicts {line_no, order_id, sku, qty}, order_id uppercased
      rejected: list of (line_no, reason)
  inventory.allocate(lines, stock) -> list of dicts with added "filled", "backorder"
  pricing.price(allocated_lines, catalog) -> list of order dicts:
      {order_id, lines: [{sku, name, filled, backorder, unit_price, line_total}],
       subtotal, tax, shipping, total}   (money as Decimal)
"""

from decimal import Decimal, ROUND_HALF_UP

from orders import parser, inventory, pricing


def _money(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _render_line(line):
    text = (
        f"  {line['sku']} {line['name']} x{line['filled']} "
        f"@ {_money(line['unit_price'])} = {_money(line['line_total'])}"
    )
    if line.get("backorder", 0) > 0:
        text += f" (backorder {line['backorder']})"
    return text


def _render_order(order):
    out = [f"ORDER {order['order_id']}"]
    out.extend(_render_line(line) for line in order["lines"])
    out.append(f"  subtotal {_money(order['subtotal'])}")
    out.append(f"  tax {_money(order['tax'])}")
    out.append(f"  shipping {_money(order['shipping'])}")
    out.append(f"  total {_money(order['total'])}")
    return "\n".join(out)


def _render_rejected(rejected):
    out = ["REJECTED"]
    out.extend(f"  line {n}: {reason}" for n, reason in sorted(rejected))
    return "\n".join(out)


def process(orders_csv, catalog, stock):
    lines, rejected = parser.parse(orders_csv)
    allocated = inventory.allocate(lines, stock)
    orders = pricing.price(allocated, catalog)

    blocks = []
    for order in sorted(orders, key=lambda o: o["order_id"]):
        if order["lines"]:
            blocks.append(_render_order(order))
    if rejected:
        blocks.append(_render_rejected(rejected))
    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
