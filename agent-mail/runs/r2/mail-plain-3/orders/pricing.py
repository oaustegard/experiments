"""Pricing for orders: line totals, order subtotal, tax, shipping, total.

All money is exact Decimal, rounded ROUND_HALF_UP to cents.

Interface (used by report.py):

    price_line(sku, filled, catalog) -> Decimal
        Line total for `filled` units of `sku` (0 allowed).

    price_order(lines, catalog) -> dict
        `lines` = the allocated lines of ONE order, as dicts with "sku" and
        "filled" (carol's allocate() output; extra keys are ignored).
        Returns:
            {
              "lines":    [ {"sku", "filled", "unit_price", "line_total"}, ... ],
              "subtotal": Decimal,
              "tax":      Decimal,
              "shipping": Decimal,
              "total":    Decimal,
            }
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
BULK_THRESHOLD = 10
BULK_FACTOR = Decimal("0.90")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_LIMIT = Decimal("50.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    return Decimal(str(catalog[sku]["price"]))


def price_line(sku, filled, catalog):
    unit = _unit_price(sku, catalog)
    total = unit * int(filled)
    if int(filled) >= BULK_THRESHOLD:
        total = total * BULK_FACTOR
    return _cents(total)


def price_order(lines, catalog):
    priced = []
    subtotal = Decimal("0.00")
    for line in lines:
        sku, filled = line["sku"], int(line["filled"])
        unit = _unit_price(sku, catalog)
        line_total = price_line(sku, filled, catalog)
        priced.append({
            "sku": sku,
            "filled": filled,
            "unit_price": _cents(unit),
            "line_total": line_total,
        })
        subtotal += line_total
    subtotal = _cents(subtotal)
    tax = _cents(subtotal * TAX_RATE)
    shipping = SHIPPING_FEE if Decimal("0") < subtotal < SHIPPING_LIMIT else _cents(0)
    shipping = _cents(shipping)
    total = _cents(subtotal + tax + shipping)
    return {
        "lines": priced,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
