"""Pricing for the orders package.

All money is exact Decimal, rounded ROUND_HALF_UP to cents. Public values
crossing module boundaries are 2-decimal strings.
"""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
FREE_SHIPPING_AT = Decimal("50.00")
SHIPPING_FEE = Decimal("5.00")


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def price_lines(lines, catalog):
    """lines: list[dict order:str sku:str filled:int] (other keys ignored).
    catalog: {sku: {"name": str, "price": str}}.
    Returns list[dict order:str sku:str unit:str total:str], in input order.
    Lines with filled 0 are priced (total 0.00). Unknown sku raises KeyError.
    """
    priced = []
    for line in lines:
        unit = _cents(catalog[line["sku"]]["price"])
        total = Decimal(line["filled"]) * unit
        if line["filled"] >= BULK_QTY:
            total = total * BULK_FACTOR
        total = _cents(total)
        priced.append({
            "order": line["order"],
            "sku": line["sku"],
            "unit": str(unit),
            "total": str(total),
        })
    return priced


def summarize(lines):
    """lines: list[dict total:str] for ONE order (extra keys ignored).
    Returns dict subtotal:str tax:str shipping:str total:str.
    """
    subtotal = sum((Decimal(line["total"]) for line in lines), Decimal("0"))
    subtotal = _cents(subtotal)
    tax = _cents(subtotal * TAX_RATE)
    if Decimal("0") < subtotal < FREE_SHIPPING_AT:
        shipping = _cents(SHIPPING_FEE)
    else:
        shipping = _cents(Decimal("0"))
    total = _cents(subtotal + tax + shipping)
    return {
        "subtotal": str(subtotal),
        "tax": str(tax),
        "shipping": str(shipping),
        "total": str(total),
    }
