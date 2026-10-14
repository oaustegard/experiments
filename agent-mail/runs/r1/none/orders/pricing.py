"""Pricing for order lines and orders. All money is exact Decimal, rounded to cents.

Interface (for alice, carol, dave):

    price_line(sku: str, filled: int, catalog: dict) -> Decimal
        Line total for `filled` units of `sku`. `catalog[sku]["price"]` is a
        decimal string. Returns 0.00 when filled == 0.

    price_order(lines: list[tuple[str, int]], catalog: dict) -> dict
        `lines` is a list of (sku, filled) pairs, one per order line, using
        FILLED quantity. Returns a dict with Decimal values:
            {"lines": [Decimal, ...],   # line totals, same order as `lines`
             "subtotal": Decimal,
             "tax": Decimal,
             "shipping": Decimal,
             "total": Decimal}
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_THRESHOLD = 10
BULK_MULTIPLIER = Decimal("0.90")
FREE_SHIPPING_AT = Decimal("50.00")
SHIPPING_FEE = Decimal("5.00")


def _cents(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def price_line(sku: str, filled: int, catalog: dict) -> Decimal:
    if filled <= 0:
        return _cents(Decimal("0"))
    unit = Decimal(catalog[sku]["price"])
    total = unit * filled
    if filled >= BULK_THRESHOLD:
        total = total * BULK_MULTIPLIER
    return _cents(total)


def price_order(lines: list, catalog: dict) -> dict:
    line_totals = [price_line(sku, filled, catalog) for sku, filled in lines]
    subtotal = _cents(sum(line_totals, Decimal("0")))
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < FREE_SHIPPING_AT:
        shipping = _cents(SHIPPING_FEE)
    else:
        shipping = _cents(Decimal("0"))
    total = _cents(subtotal + tax + shipping)
    return {
        "lines": line_totals,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
