from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
DISCOUNT_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_THRESHOLD = Decimal("50.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_line(sku, filled, catalog):
    """Return {'unit_price': Decimal, 'line_total': Decimal} for a filled quantity.

    SKUs starting with 'GIFT-' are free (unit price 0.00).
    Quantities of 10 or more get a 10% discount on the line total.
    """
    if sku.startswith("GIFT-"):
        unit = Decimal("0.00")
    else:
        unit = _cents(catalog[sku]["price"])
    total = Decimal(filled) * unit
    if filled >= 10:
        total = total * DISCOUNT_FACTOR
    return {"unit_price": unit, "line_total": _cents(total)}


def price_order(line_totals):
    """Return {'subtotal','tax','shipping','total'} as Decimals, rounded to cents."""
    subtotal = _cents(sum((_cents(t) for t in line_totals), Decimal("0")))
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < SHIPPING_THRESHOLD:
        shipping = _cents(SHIPPING_FEE)
    else:
        shipping = _cents(Decimal("0"))
    total = _cents(subtotal + tax + shipping)
    return {"subtotal": subtotal, "tax": tax, "shipping": shipping, "total": total}
