from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_FACTOR = Decimal("0.90")


def _q(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_line(sku, filled, catalog):
    """Line total for `filled` units of `sku`, as a 2dp string."""
    if sku.startswith("GIFT-"):
        return "0.00"
    unit = Decimal(catalog[sku]["price"])
    total = unit * filled
    if filled >= 10:
        total = total * BULK_FACTOR
    return str(_q(total))


def price_order(lines, catalog):
    """Price an order from its filled lines.

    lines: list of dicts with keys sku, filled.
    Returns dict with subtotal, tax, shipping, total (2dp strings).
    """
    subtotal = Decimal("0.00")
    for line in lines:
        subtotal += Decimal(price_line(line["sku"], line["filled"], catalog))
    subtotal = _q(subtotal)
    tax = _q(subtotal * TAX_RATE)
    shipping = _q("5.00") if Decimal("0") < subtotal < Decimal("50.00") else _q("0.00")
    total = _q(subtotal + tax + shipping)
    return {
        "subtotal": str(subtotal),
        "tax": str(tax),
        "shipping": str(shipping),
        "total": str(total),
    }
