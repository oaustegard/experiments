"""Entry point: run the order pipeline and render the invoice text."""

from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse_orders
from orders.inventory import allocate
from orders.pricing import price_line, price_order

CENT = Decimal("0.01")


def _money(amount):
    return Decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP)


def _format_money(amount):
    return f"{_money(amount):.2f}"


def process(orders_csv, catalog, stock):
    lines, rejections = parse_orders(orders_csv)
    rejected = [(r["line_no"], r["reason"]) for r in rejections]

    # SKUs missing from the catalog cannot be priced; reject them up front.
    priceable = []
    for line in lines:
        if line["sku"] in catalog:
            priceable.append(line)
        else:
            rejected.append((line["line_no"], "unknown sku"))

    allocated, alloc_rejected = allocate(priceable, stock)
    rejected.extend(alloc_rejected)
    rejected.sort(key=lambda item: item[0])

    # Group accepted lines by order, orders in plain string order,
    # lines in input order (allocated is already sorted by line number).
    by_order = {}
    for line in allocated:
        by_order.setdefault(line["order_id"], []).append(line)

    blocks = []
    for order_id in sorted(by_order):
        order_lines = by_order[order_id]
        out = [f"ORDER {order_id}"]
        for line in order_lines:
            sku = line["sku"]
            entry = catalog[sku]
            filled = line["filled"]
            unit = _format_money(entry["price"])
            total = _format_money(price_line(sku, filled, catalog))
            text = f"  {sku} {entry['name']} x{filled} @ {unit} = {total}"
            if line["backorder"] > 0:
                text += f" (backorder {line['backorder']})"
            out.append(text)
        totals = price_order(
            [(line["sku"], line["filled"]) for line in order_lines], catalog
        )
        for key in ("subtotal", "tax", "shipping", "total"):
            out.append(f"  {key} {_format_money(totals[key])}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for line_no, reason in rejected:
            out.append(f"  line {line_no}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
