from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def price_orders(filled_lines, catalog):
    """Price filled lines per order.

    filled_lines: list of dict with n, order, sku, filled (int).
    catalog: {sku: {"name": str, "price": "12.50"}}.
    Returns one dict per order, in first-seen order, with money as 2-place strings.
    """
    grouped = {}
    for line in filled_lines:
        grouped.setdefault(line["order"], []).append(line)

    result = []
    for order_id, lines in grouped.items():
        priced_lines = []
        subtotal = Decimal("0")
        for line in lines:
            if line["sku"].startswith("GIFT-"):
                unit = Decimal("0.00")
            else:
                unit = _cents(Decimal(catalog[line["sku"]]["price"]))
            filled = line["filled"]
            total = Decimal(filled) * unit
            if filled >= BULK_QTY:
                total = total * BULK_FACTOR
            total = _cents(total)
            subtotal += total
            priced_lines.append({
                "n": line["n"],
                "sku": line["sku"],
                "filled": filled,
                "unit": str(unit),
                "total": str(total),
            })

        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        if Decimal("0") < subtotal < FREE_SHIPPING_AT:
            shipping = _cents(SHIPPING_FEE)
        else:
            shipping = _cents(Decimal("0"))
        total = subtotal + tax + shipping

        result.append({
            "order": order_id,
            "lines": priced_lines,
            "subtotal": str(subtotal),
            "tax": str(tax),
            "shipping": str(shipping),
            "total": str(_cents(total)),
        })
    return result
