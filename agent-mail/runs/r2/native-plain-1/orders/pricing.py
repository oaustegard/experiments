"""Pricing for order lines and orders. All money is exact Decimal, rounded to cents (ROUND_HALF_UP).

Interface:
    price_line(catalog, sku, filled) -> Decimal
        Line total for `filled` units of `sku`. `filled` is the FILLED quantity.

    price_order(catalog, lines) -> dict
        `lines` is a list of (sku, filled) tuples, in order.
        Returns {"lines": [Decimal, ...], "subtotal": Decimal, "tax": Decimal,
                 "shipping": Decimal, "total": Decimal}.
"""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
FREE_SHIPPING_AT = Decimal("50.00")
SHIPPING_FEE = Decimal("5.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_line(catalog, sku, filled):
    unit = Decimal(catalog[sku]["price"])
    total = unit * int(filled)
    if int(filled) >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price_order(catalog, lines):
    line_totals = [price_line(catalog, sku, filled) for sku, filled in lines]
    subtotal = _cents(sum(line_totals, Decimal("0")))
    tax = _cents(subtotal * TAX_RATE)
    shipping = SHIPPING_FEE if Decimal("0") < subtotal < FREE_SHIPPING_AT else Decimal("0.00")
    shipping = _cents(shipping)
    total = _cents(subtotal + tax + shipping)
    return {
        "lines": line_totals,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
