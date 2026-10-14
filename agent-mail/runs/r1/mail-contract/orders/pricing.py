"""Pricing for the orders package. Money is exact Decimal, rounded HALF_UP to cents."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_THRESHOLD = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
SHIPPING_LIMIT = Decimal("50.00")
ZERO = Decimal("0.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_orders(lines, catalog):
    """Price every line and group the results by order.

    lines:   list of dicts with keys n:int, order:str, sku:str, filled:int
             (filled is the allocated quantity, not the ordered one; extra keys are ignored)
    catalog: {sku: {"name": str, "price": "12.50"}}
    returns: {order: {"lines": [{"n", "sku", "filled", "unit_price", "line_total"}],
                      "subtotal", "tax", "shipping", "total"}}
             orders in first-appearance order; money values are Decimal quantized to 0.01.
    """
    orders = {}
    for line in lines:
        unit_price = Decimal(catalog[line["sku"]]["price"])
        filled = int(line["filled"])
        line_total = unit_price * filled
        if filled >= BULK_THRESHOLD:
            line_total = line_total * BULK_FACTOR
        priced = {
            "n": line["n"],
            "sku": line["sku"],
            "filled": filled,
            "unit_price": _cents(unit_price),
            "line_total": _cents(line_total),
        }
        order = orders.setdefault(line["order"], {"lines": [], "subtotal": ZERO})
        order["lines"].append(priced)
        order["subtotal"] = order["subtotal"] + priced["line_total"]

    for order in orders.values():
        subtotal = _cents(order["subtotal"])
        tax = _cents(subtotal * TAX_RATE)
        shipping = SHIPPING_FEE if ZERO < subtotal < SHIPPING_LIMIT else ZERO
        shipping = _cents(shipping)
        order["subtotal"] = subtotal
        order["tax"] = tax
        order["shipping"] = shipping
        order["total"] = _cents(subtotal + tax + shipping)
    return orders
