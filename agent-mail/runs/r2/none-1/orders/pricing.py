"""Pricing for orders. All money is exact Decimal, rounded ROUND_HALF_UP to cents.

Interface (bob's side, consumed by report.py):

    price_line(sku, filled, catalog) -> Decimal
        Line total for `filled` units of `sku`. 10 or more units get 10% off.

    price_order(filled_lines, catalog) -> dict
        `filled_lines` is an iterable of (sku, filled) pairs, filled > 0.
        Returns {"lines": [(sku, filled, line_total), ...],
                 "subtotal": Decimal, "tax": Decimal,
                 "shipping": Decimal, "total": Decimal}
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_THRESHOLD = 10
BULK_FACTOR = Decimal("0.90")
FREE_SHIPPING_AT = Decimal("50.00")
SHIPPING_FEE = Decimal("5.00")


def _money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def unit_price(sku, catalog):
    return Decimal(catalog[sku]["price"])


def price_line(sku, filled, catalog):
    total = unit_price(sku, catalog) * filled
    if filled >= BULK_THRESHOLD:
        total = total * BULK_FACTOR
    return _money(total)


def price_order(filled_lines, catalog):
    lines = []
    for sku, filled in filled_lines:
        lines.append((sku, filled, price_line(sku, filled, catalog)))

    subtotal = _money(sum((t for _, _, t in lines), Decimal("0")))
    tax = _money(subtotal * TAX_RATE)
    shipping = SHIPPING_FEE if Decimal("0") < subtotal < FREE_SHIPPING_AT else Decimal("0.00")
    shipping = _money(shipping)
    total = _money(subtotal + tax + shipping)

    return {
        "lines": lines,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
