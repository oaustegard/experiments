"""Pricing for orders: exact decimal arithmetic, rounded half-up to cents."""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_THRESHOLD = Decimal("50.00")
ZERO = Decimal("0.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _line_total(unit_price, filled):
    """filled * unit price, with the bulk discount at 10+ units, rounded to cents."""
    if filled <= 0:
        return ZERO
    total = unit_price * filled
    if filled >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price(lines, catalog):
    """Price allocated order lines.

    lines: list of dicts with "order", "sku", "filled" (as from
        inventory.allocate; "order" is the order key used by parser and carol). Other keys are kept.
    catalog: {sku: {"name": str, "price": "12.50"}}

    Returns (priced_lines, totals):
        priced_lines: new list of dicts, input keys plus "unit_price" and
            "line_total" (Decimal, cents). Pricing uses "filled", not "qty".
        totals: {order: {"subtotal", "tax", "shipping", "total"}}, all
            Decimal rounded to cents. Orders with no lines are not present.
    """
    priced_lines = []
    subtotals = {}
    for entry in lines:
        unit_price = Decimal(catalog[entry["sku"]]["price"])
        filled = int(entry["filled"])
        line_total = _line_total(unit_price, filled)
        priced = dict(entry)
        priced["unit_price"] = _cents(unit_price)
        priced["line_total"] = line_total
        priced_lines.append(priced)
        order = entry["order"]
        subtotals[order] = subtotals.get(order, ZERO) + line_total

    totals = {}
    for order, raw_subtotal in subtotals.items():
        subtotal = _cents(raw_subtotal)
        tax = _cents(subtotal * TAX_RATE)
        if ZERO < subtotal < SHIPPING_THRESHOLD:
            shipping = _cents(SHIPPING_FEE)
        else:
            shipping = ZERO
        totals[order] = {
            "subtotal": subtotal,
            "tax": tax,
            "shipping": shipping,
            "total": _cents(subtotal + tax + shipping),
        }
    return priced_lines, totals
