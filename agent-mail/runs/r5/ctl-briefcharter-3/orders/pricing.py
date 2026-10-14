from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_FACTOR = Decimal("0.90")
SHIPPING = Decimal("5.00")


def _money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_orders(lines, catalog):
    """Price filled lines grouped by order.

    lines: list of dict n, order, sku, filled (filled quantity, not ordered).
    catalog: {sku: {"name": str, "price": "12.50"}}.
    Returns a list (first-appearance order) of dicts:
      order, lines (list of n, sku, filled, unit, total), subtotal, tax, shipping, total.
    Money fields are 2-dp strings.
    """
    grouped = {}
    for line in lines:
        grouped.setdefault(line["order"], []).append(line)

    result = []
    for order, order_lines in grouped.items():
        priced_lines = []
        subtotal = Decimal("0")
        for line in order_lines:
            if line["sku"].startswith("GIFT-"):
                unit = Decimal("0.00")
            else:
                unit = _money(catalog[line["sku"]]["price"])
            total = unit * line["filled"]
            if line["filled"] >= 10:
                total = total * BULK_FACTOR
            total = _money(total)
            subtotal += total
            priced_lines.append({
                "n": line["n"],
                "sku": line["sku"],
                "filled": line["filled"],
                "unit": str(unit),
                "total": str(total),
            })
        subtotal = _money(subtotal)
        tax = _money(subtotal * TAX_RATE)
        shipping = SHIPPING if Decimal("0") < subtotal < Decimal("50.00") else Decimal("0.00")
        total = subtotal + tax + shipping
        result.append({
            "order": order,
            "lines": priced_lines,
            "subtotal": str(subtotal),
            "tax": str(tax),
            "shipping": str(_money(shipping)),
            "total": str(_money(total)),
        })
    return result
