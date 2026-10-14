"""Price allocated order lines and roll them up per order.

Contract:
    price(allocated: list[dict n:int order:str sku:str filled:int],
          catalog: dict[str, dict]) -> (priced, orders)
        priced: list[dict n:int order:str sku:str filled:int unit:str total:str]
        orders: list[dict order:str subtotal:str tax:str shipping:str total:str]
    All money is a 2-dp decimal string. Orders appear in first-appearance order.
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
BULK_QTY = 10
BULK_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
FREE_SHIPPING_AT = Decimal("50.00")


def _cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _money(value: Decimal) -> str:
    return str(_cents(value))


def _line_total(filled: int, unit: Decimal) -> Decimal:
    total = Decimal(filled) * unit
    if filled >= BULK_QTY:
        total = total * BULK_FACTOR
    return _cents(total)


def _shipping(subtotal: Decimal) -> Decimal:
    if Decimal("0") < subtotal < FREE_SHIPPING_AT:
        return _cents(SHIPPING_FEE)
    return _cents(Decimal("0"))


def price(allocated, catalog):
    priced = []
    orders = {}  # order id -> running subtotal, insertion-ordered

    for line in allocated:
        unit = _cents(Decimal(catalog[line["sku"]]["price"]))
        total = _line_total(line["filled"], unit)
        priced.append({
            "n": line["n"],
            "order": line["order"],
            "sku": line["sku"],
            "filled": line["filled"],
            "unit": _money(unit),
            "total": _money(total),
        })
        orders[line["order"]] = orders.get(line["order"], Decimal("0")) + total

    summaries = []
    for order_id, subtotal in orders.items():
        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        shipping = _shipping(subtotal)
        total = subtotal + tax + shipping
        summaries.append({
            "order": order_id,
            "subtotal": _money(subtotal),
            "tax": _money(tax),
            "shipping": _money(shipping),
            "total": _money(total),
        })

    return priced, summaries
