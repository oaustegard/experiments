"""Price order lines and orders. Money is exact Decimal, rounded HALF_UP to cents."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")
ZERO = Decimal("0.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_orders(lines, catalog):
    """lines: list of dicts with order, sku, filled (int).
    catalog: {sku: {"name": str, "price": "12.50"}}.
    Returns {order: {"lines": [...], "subtotal", "tax", "shipping", "total"}}
    where lines are {"sku", "filled", "unit", "total"} and money is a 2dp string.
    """
    grouped = {}
    for line in lines:
        grouped.setdefault(line["order"], []).append(line)

    result = {}
    for order, order_lines in grouped.items():
        priced = []
        subtotal = ZERO
        for line in order_lines:
            unit = Decimal(catalog[line["sku"]]["price"])
            filled = line["filled"]
            total = unit * filled
            if filled >= BULK_QTY:
                total *= BULK_FACTOR
            total = _cents(total)
            subtotal += total
            priced.append({
                "sku": line["sku"],
                "filled": filled,
                "unit": _cents(unit),
                "total": total,
            })

        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        shipping = SHIPPING_FEE if ZERO < subtotal < FREE_SHIPPING_AT else ZERO
        total = subtotal + tax + shipping

        result[order] = {
            "lines": [
                {"sku": p["sku"], "filled": p["filled"],
                 "unit": str(p["unit"]), "total": str(p["total"])}
                for p in priced
            ],
            "subtotal": str(subtotal),
            "tax": str(tax),
            "shipping": str(_cents(shipping)),
            "total": str(_cents(total)),
        }
    return result
