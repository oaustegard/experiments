from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")
ZERO = Decimal("0.00")


def _q(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price(lines, catalog):
    """Price each line by its FILLED quantity.

    Input: list of dicts with 'filled' (int) and 'sku' (str); catalog as spec.
    Returns a new list in the same order; each dict gets
    unit (Decimal 2dp) and total (Decimal 2dp, 0.00 if filled is 0).
    Any sku starting with 'GIFT-' has unit price 0.00.
    """
    out = []
    for line in lines:
        sku = line["sku"]
        filled = line["filled"]
        if str(sku).startswith("GIFT-"):
            unit = ZERO
        else:
            unit = _q(Decimal(catalog[sku]["price"]))
        total = unit * filled
        if filled >= 10:
            total = total * Decimal("0.90")
        total = _q(total) if filled else ZERO
        new = dict(line)
        new["unit"] = unit
        new["total"] = total
        out.append(new)
    return out


def totals(lines):
    """Order totals from priced lines (each with 'total' Decimal).

    Returns dict subtotal, tax, shipping, total (Decimal 2dp, ROUND_HALF_UP).
    """
    subtotal = _q(sum((l["total"] for l in lines), ZERO))
    tax = _q(subtotal * TAX_RATE)
    if ZERO < subtotal < FREE_SHIPPING_AT:
        shipping = _q(SHIPPING_FEE)
    else:
        shipping = ZERO
    total = _q(subtotal + tax + shipping)
    return {"subtotal": subtotal, "tax": tax, "shipping": shipping, "total": total}
