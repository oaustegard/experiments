"""Pricing for orders.

Provides:
    price_orders(lines:list[dict n:int order:str sku:str filled:int],
                 catalog:dict) -> dict[order:str -> dict subtotal:str tax:str
                 shipping:str total:str lines:list[dict n:int sku:str filled:int
                 line_total:str]]

All money is exact Decimal, rounded ROUND_HALF_UP to cents.
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_FACTOR = Decimal("0.90")
BULK_MIN_QTY = 10
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_THRESHOLD = Decimal("50.00")
GIFT_PREFIX = "GIFT-"


def _cents(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    # Rule from alice: any GIFT- sku is free, whatever the catalog says.
    if sku.startswith(GIFT_PREFIX):
        return Decimal("0.00")
    return Decimal(catalog[sku]["price"])


def _line_total(filled, unit):
    total = Decimal(filled) * unit
    if filled >= BULK_MIN_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def price_orders(lines, catalog):
    grouped = {}
    for line in lines:
        grouped.setdefault(line["order"], []).append(line)

    result = {}
    for order, order_lines in grouped.items():
        priced = []
        subtotal = Decimal("0.00")
        for line in order_lines:
            filled = line["filled"]
            line_total = _line_total(filled, _unit_price(line["sku"], catalog))
            subtotal += line_total
            priced.append({
                "n": line["n"],
                "sku": line["sku"],
                "filled": filled,
                "line_total": f"{line_total:.2f}",
            })

        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        if Decimal("0") < subtotal < FREE_SHIPPING_THRESHOLD:
            shipping = SHIPPING_FEE
        else:
            shipping = Decimal("0.00")
        total = subtotal + tax + shipping

        result[order] = {
            "subtotal": f"{subtotal:.2f}",
            "tax": f"{tax:.2f}",
            "shipping": f"{shipping:.2f}",
            "total": f"{total:.2f}",
            "lines": priced,
        }
    return result
