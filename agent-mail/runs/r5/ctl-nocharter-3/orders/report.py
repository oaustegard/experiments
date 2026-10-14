from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_orders


def _money(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def process(orders_csv, catalog, stock):
    lines, parse_rejected = parse(orders_csv)
    allocated, alloc_rejected = allocate(lines, stock)
    rejected = sorted(list(parse_rejected) + list(alloc_rejected), key=lambda r: r[0])

    # Lines with nothing filled are not priced; they are printed as 0.00 below.
    priced = price_orders([l for l in allocated if l["filled"] > 0], catalog)

    by_order = {}
    for line in allocated:
        by_order.setdefault(line["order"], []).append(line)

    blocks = []
    for order_id in sorted(by_order):
        priced_order = priced.get(order_id)
        priced_lines = {pl["sku"]: pl for pl in priced_order["lines"]} if priced_order else {}

        out = [f"ORDER {order_id}"]
        for line in by_order[order_id]:
            sku = line["sku"]
            filled = line["filled"]
            name = catalog[sku]["name"]
            if filled > 0:
                pl = priced_lines[sku]
                unit, total = pl["unit"], pl["total"]
            else:
                unit, total = _money(catalog[sku]["price"]), "0.00"
            text = f"  {sku} {name} x{filled} @ {unit} = {total}"
            if line["backorder"] > 0:
                text += f" (backorder {line['backorder']})"
            out.append(text)

        if priced_order:
            subtotal, tax = priced_order["subtotal"], priced_order["tax"]
            shipping, total = priced_order["shipping"], priced_order["total"]
        else:
            subtotal = tax = shipping = total = "0.00"
        out.append(f"  subtotal {subtotal}")
        out.append(f"  tax {tax}")
        out.append(f"  shipping {shipping}")
        out.append(f"  total {total}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
