"""Price allocated order lines and orders with exact decimal money."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _line_total(filled, unit_price):
    total = filled * unit_price
    if filled >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price_orders(allocs, catalog):
    """Price allocated lines grouped by order, in first-seen order.

    allocs: list of dicts with order, sku and filled (units actually filled).
    catalog: {sku: {"name": str, "price": "12.50"}}.
    Zero-filled lines are kept with total 0.00; orders are never dropped.
    Returns a list of dicts: order, lines (sku, filled, total), subtotal,
    tax, shipping, total.
    """
    grouped = {}
    for alloc in allocs:
        order_lines = grouped.setdefault(alloc["order"], [])
        filled = alloc["filled"]
        sku = alloc["sku"]
        if filled <= 0:
            unit_price = Decimal("0.00")
        elif sku.startswith("GIFT-"):
            unit_price = Decimal("0.00")
        else:
            unit_price = Decimal(catalog[sku]["price"])
        order_lines.append({
            "sku": sku,
            "filled": filled,
            "total": _line_total(filled, unit_price),
        })

    invoices = []
    for order, lines in grouped.items():
        subtotal = sum((line["total"] for line in lines), Decimal("0.00"))
        tax = _cents(subtotal * TAX_RATE)
        if Decimal("0") < subtotal < FREE_SHIPPING_AT:
            shipping = SHIPPING_FEE
        else:
            shipping = Decimal("0.00")
        total = subtotal + tax + shipping
        invoices.append({
            "order": order,
            "lines": lines,
            "subtotal": subtotal,
            "tax": tax,
            "shipping": shipping,
            "total": total,
        })
    return invoices
