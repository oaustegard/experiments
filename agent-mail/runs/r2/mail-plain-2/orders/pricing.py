"""Pricing for order lines and whole orders. Exact decimal, ROUND_HALF_UP to cents."""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_FROM = Decimal("50.00")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def unit_price(sku, catalog):
    """Decimal unit price of sku, from catalog[sku]["price"]."""
    return Decimal(catalog[sku]["price"])


def line_total(sku, filled, catalog):
    """Price of `filled` units of sku, rounded to cents. Uses FILLED quantity."""
    total = unit_price(sku, catalog) * filled
    if filled >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price_order(lines, catalog):
    """Price an order.

    lines: one order's lines, each either a (sku, filled) tuple or a dict with
           "sku" and "filled" keys (extra keys are ignored). filled = units shipped.
    Returns a dict:
      "lines":    input dicts (copied) plus "name", "unit" (Decimal unit price),
                  "amount" (Decimal line total, cents); tuples become sku/filled dicts
      "subtotal", "tax", "shipping", "total": Decimals rounded to cents
    """
    priced = []
    for line in lines:
        if isinstance(line, dict):
            sku, filled = line["sku"], line["filled"]
            out = dict(line)
        else:
            sku, filled = line
            out = {"sku": sku, "filled": filled}
        out["name"] = catalog[sku]["name"]
        out["unit"] = unit_price(sku, catalog)
        out["amount"] = line_total(sku, filled, catalog)
        priced.append(out)
    subtotal = sum((p["amount"] for p in priced), Decimal("0.00"))
    tax = _cents(subtotal * TAX_RATE)
    shipping = SHIPPING_FEE if Decimal("0") < subtotal < FREE_SHIPPING_FROM else Decimal("0.00")
    total = subtotal + tax + shipping
    return {
        "lines": priced,
        "subtotal": subtotal,
        "tax": tax,
        "shipping": shipping,
        "total": total,
    }
