"""Pricing for allocated order lines.

All money is exact Decimal, rounded ROUND_HALF_UP to cents.

Interface (agreed for the team):

    price_line(sku, filled, catalog) -> Decimal
        Line total for `filled` units of `sku`. Applies the 10+ unit 10% discount.

    price_order(lines, catalog) -> dict
        `lines` is a list of allocated lines, each a dict with keys
        "sku" (str) and "filled" (int). Returns:
            {
              "lines":    [{"sku", "filled", "unit_price", "line_total"}, ...],
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
BULK_MULTIPLIER = Decimal("0.90")
FREE_SHIPPING_SUBTOTAL = Decimal("50.00")
SHIPPING_FEE = Decimal("5.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    return Decimal(str(catalog[sku]["price"]))


def price_line(sku, filled, catalog):
    """Line total for `filled` units of `sku`, rounded to cents."""
    filled = int(filled)
    total = _unit_price(sku, catalog) * filled
    if filled >= BULK_THRESHOLD:
        total = total * BULK_MULTIPLIER
    return _cents(total)


def price_order(lines, catalog):
    """Price a list of allocated lines (each {"sku", "filled"}).

    Uses FILLED quantity only. Returns a dict of line details and order totals.
    """
    priced = []
    subtotal = Decimal("0.00")
    for line in lines:
        sku = line["sku"]
        filled = int(line["filled"])
        line_total = price_line(sku, filled, catalog)
        subtotal += line_total
        priced.append({
            "sku": sku,
            "filled": filled,
            "unit_price": _cents(_unit_price(sku, catalog)),
            "line_total": line_total,
        })

    subtotal = _cents(subtotal)
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < FREE_SHIPPING_SUBTOTAL:
        shipping = _cents(SHIPPING_FEE)
    else:
        shipping = _cents(Decimal("0"))
    total = _cents(subtotal + tax + shipping)

    return {
        "lines": priced,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
