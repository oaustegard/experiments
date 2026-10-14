from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_THRESHOLD = Decimal("50.00")


def _cents(amount):
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def _line_total(filled, unit):
    total = Decimal(filled) * unit
    if filled >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price_orders(lines, catalog):
    """Price allocated lines, grouped by order id (first-appearance order)."""
    grouped = {}
    for line in lines:
        grouped.setdefault(line["order"], []).append(line)

    result = {}
    for order_id, order_lines in grouped.items():
        priced = []
        subtotal = Decimal("0")
        for line in order_lines:
            unit_str = catalog[line["sku"]]["price"]
            total = _line_total(line["filled"], Decimal(unit_str))
            subtotal += total
            priced.append({
                "n": line["n"],
                "sku": line["sku"],
                "filled": line["filled"],
                "unit": unit_str,
                "total": str(total),
            })
        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        if Decimal("0") < subtotal < SHIPPING_THRESHOLD:
            shipping = _cents(SHIPPING_FEE)
        else:
            shipping = _cents(Decimal("0"))
        total = _cents(subtotal + tax + shipping)
        result[order_id] = {
            "subtotal": str(subtotal),
            "tax": str(tax),
            "shipping": str(shipping),
            "total": str(total),
            "lines": priced,
        }
    return result
