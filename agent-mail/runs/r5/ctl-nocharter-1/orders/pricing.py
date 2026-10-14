from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")
ZERO = Decimal("0.00")


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def price_orders(allocated, catalog):
    """Price allocated order lines.

    allocated: {order: [{"sku": str, "filled": int}, ...]}, orders in first-appearance order
    catalog:   {sku: {"name": str, "price": "12.50"}}
    returns:   [{"order", "lines": [{"sku", "name", "filled", "unit", "total"}],
                 "subtotal", "tax", "shipping", "total"}, ...]  (Decimal money)
    """
    priced = []
    for order, lines in allocated.items():
        priced_lines = []
        subtotal = ZERO
        for line in lines:
            sku = line["sku"]
            filled = line["filled"]
            item = catalog[sku]
            unit = Decimal(item["price"])
            total = filled * unit
            if filled >= BULK_QTY:
                total = total * BULK_FACTOR
            total = _cents(total)
            subtotal += total
            priced_lines.append({
                "sku": sku,
                "name": item["name"],
                "filled": filled,
                "unit": unit,
                "total": total,
            })

        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        shipping = SHIPPING_FEE if ZERO < subtotal < FREE_SHIPPING_AT else ZERO
        total = subtotal + tax + shipping
        priced.append({
            "order": order,
            "lines": priced_lines,
            "subtotal": subtotal,
            "tax": tax,
            "shipping": shipping,
            "total": total,
        })
    return priced
