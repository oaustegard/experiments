from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
TAX_RATE = Decimal("0.0825")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_THRESHOLD = Decimal("50.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price(allocations, catalog):
    """Price allocated lines. Returns new dicts with name, unit_price, line_total.

    Pricing uses the filled quantity. Bulk discount applies at filled >= 10.
    """
    priced = []
    for alloc in allocations:
        entry = dict(alloc)
        item = catalog[alloc["sku"]]
        unit_price = Decimal(item["price"])
        line_total = Decimal(alloc["filled"]) * unit_price
        if alloc["filled"] >= BULK_QTY:
            line_total = line_total * BULK_FACTOR
        entry["name"] = item["name"]
        entry["unit_price"] = unit_price
        entry["line_total"] = _cents(line_total)
        priced.append(entry)
    return priced


def order_totals(priced_lines):
    """Compute subtotal, tax, shipping and total for one order, all at 0.01."""
    subtotal = _cents(sum((Decimal(l["line_total"]) for l in priced_lines), Decimal("0")))
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < FREE_SHIPPING_THRESHOLD:
        shipping = _cents(SHIPPING_FEE)
    else:
        shipping = _cents(Decimal("0"))
    total = _cents(subtotal + tax + shipping)
    return {"subtotal": subtotal, "tax": tax, "shipping": shipping, "total": total}
