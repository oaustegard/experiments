from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
SHIPPING = Decimal("5.00")
FREE_SHIPPING_FROM = Decimal("50.00")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    # Rule from alice: SKUs starting with GIFT- are free.
    if sku.startswith("GIFT-"):
        return Decimal("0.00")
    return Decimal(catalog[sku]["price"])


def price(lines, catalog):
    priced = []
    for line in lines:
        unit = _unit_price(line["sku"], catalog)
        total = unit * line["filled"]
        if line["filled"] >= BULK_QTY:
            total = total * BULK_FACTOR
        total = _cents(total)
        out = dict(line)
        out["unit"] = str(_cents(unit))
        out["line_total"] = str(total)
        priced.append(out)
    return priced


def order_totals(priced):
    subtotals = {}
    for line in priced:
        subtotals[line["order"]] = subtotals.get(line["order"], Decimal("0")) + Decimal(line["line_total"])
    totals = {}
    for order, subtotal in subtotals.items():
        tax = _cents(subtotal * TAX_RATE)
        shipping = SHIPPING if Decimal("0") < subtotal < FREE_SHIPPING_FROM else Decimal("0.00")
        total = subtotal + tax + shipping
        totals[order] = {
            "subtotal": str(_cents(subtotal)),
            "tax": str(tax),
            "shipping": str(_cents(shipping)),
            "total": str(_cents(total)),
        }
    return totals
