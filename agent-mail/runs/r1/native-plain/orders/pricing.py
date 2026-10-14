"""Pricing for order lines and orders.

All money is exact Decimal, rounded ROUND_HALF_UP to cents.

Interface (for the other modules):

    price_line(sku: str, filled: int, catalog: dict) -> Decimal
        Line total for `filled` units of `sku`, bulk discount applied, in cents.

    price_order(lines, catalog: dict) -> dict
        `lines` is an iterable of (sku, filled) pairs, one per order line,
        where `filled` is the FILLED quantity (not ordered).
        Returns {"subtotal": Decimal, "tax": Decimal,
                 "shipping": Decimal, "total": Decimal}, all in cents.
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_THRESHOLD = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_LIMIT = Decimal("50.00")
ZERO = Decimal("0.00")


def _cents(amount):
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def unit_price(sku, catalog):
    return Decimal(catalog[sku]["price"])


def price_line(sku, filled, catalog):
    total = unit_price(sku, catalog) * filled
    if filled >= BULK_THRESHOLD:
        total = total * BULK_FACTOR
    return _cents(total)


def price_order(lines, catalog):
    subtotal = ZERO
    for sku, filled in lines:
        subtotal += price_line(sku, filled, catalog)
    subtotal = _cents(subtotal)
    tax = _cents(subtotal * TAX_RATE)
    shipping = SHIPPING_FEE if ZERO < subtotal < SHIPPING_LIMIT else ZERO
    total = _cents(subtotal + tax + shipping)
    return {
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
