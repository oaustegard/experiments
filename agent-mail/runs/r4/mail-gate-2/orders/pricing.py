"""Pricing for order lines and orders. All money is exact Decimal, rounded to cents (ROUND_HALF_UP)."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_FACTOR = Decimal("0.90")
FREE_PREFIX = "GIFT-"


def _cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _fmt(value: Decimal) -> str:
    return format(_cents(value), ".2f")


def price(alloc: list, catalog: dict) -> list:
    """Price each allocated line using FILLED quantity.

    Returns one dict per input line with unit and total as cent strings.
    """
    priced = []
    for line in alloc:
        sku = line["sku"]
        entry = catalog[sku]
        if sku.startswith(FREE_PREFIX):
            unit = Decimal("0.00")
        else:
            unit = Decimal(str(entry["price"]))
        filled = int(line["filled"])
        total = Decimal(filled) * unit
        if filled >= 10:
            total = total * BULK_FACTOR
        priced.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "name": entry["name"],
            "qty": line["qty"],
            "filled": filled,
            "backorder": line["backorder"],
            "unit": _fmt(unit),
            "total": _fmt(total),
        })
    return priced


def order_totals(priced: list) -> dict:
    """Per-order subtotal, tax, shipping and total as cent strings, keyed by order in first-seen order."""
    subtotals: dict = {}
    for line in priced:
        order = line["order"]
        subtotals[order] = subtotals.get(order, Decimal("0")) + Decimal(line["total"])

    result = {}
    for order, raw_subtotal in subtotals.items():
        subtotal = _cents(raw_subtotal)
        tax = _cents(subtotal * TAX_RATE)
        if Decimal("0") < subtotal < Decimal("50.00"):
            shipping = Decimal("5.00")
        else:
            shipping = Decimal("0.00")
        total = subtotal + tax + shipping
        result[order] = {
            "subtotal": _fmt(subtotal),
            "tax": _fmt(tax),
            "shipping": _fmt(shipping),
            "total": _fmt(total),
        }
    return result
