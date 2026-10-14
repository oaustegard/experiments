"""Pricing for the orders package.

All money is exact Decimal, rounded ROUND_HALF_UP to cents.

Interface (proposed by bob; other modules should call these):

    unit_price(catalog, sku) -> Decimal
    price_line(sku, filled, catalog) -> Decimal
        line total = filled * unit price, x0.90 if filled >= 10, rounded to cents.
    price_order(lines, catalog) -> dict
        lines: iterable of (sku, filled) pairs; filled is the FILLED quantity.
        returns {
            "lines":    [(sku, filled, line_total), ...],
            "subtotal": Decimal,
            "tax":      Decimal,
            "shipping": Decimal,
            "total":    Decimal,
        }
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_THRESHOLD = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_FROM = Decimal("50.00")


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def unit_price(catalog, sku):
    return Decimal(str(catalog[sku]["price"]))


def price_line(sku, filled, catalog):
    total = unit_price(catalog, sku) * filled
    if filled >= BULK_THRESHOLD:
        total = total * BULK_FACTOR
    return _cents(total)


def price_order(lines, catalog):
    priced = []
    subtotal = Decimal("0")
    for sku, filled in lines:
        line_total = price_line(sku, filled, catalog)
        priced.append((sku, filled, line_total))
        subtotal += line_total
    subtotal = _cents(subtotal)
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < FREE_SHIPPING_FROM:
        shipping = SHIPPING_FEE
    else:
        shipping = Decimal("0.00")
    total = subtotal + tax + shipping
    return {
        "lines": priced,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
