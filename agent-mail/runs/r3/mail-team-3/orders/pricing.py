"""Pricing for orders.

price(lines, catalog) -> (priced, orders)
  lines:   list[dict n:int order:str sku:str filled:int]
  catalog: dict[str, dict]  ({"name": str, "price": "12.50"})
  priced:  list[dict n:int order:str sku:str filled:int unit:Decimal line_total:Decimal]
  orders:  dict[str, dict order:str subtotal:Decimal tax:Decimal shipping:Decimal total:Decimal]
           (keys in first-appearance order of orders in lines)
"""

from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
TAX_RATE = Decimal("0.0825")
DISCOUNT_FACTOR = Decimal("0.90")
SHIPPING_FEE = Decimal("5.00")
FREE_GIFT_PREFIX = "GIFT-"


def _cents(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    if sku.startswith(FREE_GIFT_PREFIX):
        return Decimal("0.00")
    return _cents(Decimal(catalog[sku]["price"]))


def price(lines, catalog):
    priced = []
    orders = {}
    for line in lines:
        sku = line["sku"]
        filled = int(line["filled"])
        unit = _unit_price(sku, catalog)
        total = unit * filled
        if filled >= 10:
            total = total * DISCOUNT_FACTOR
        line_total = _cents(total)
        priced.append({
            "n": line["n"],
            "order": line["order"],
            "sku": sku,
            "filled": filled,
            "unit": unit,
            "line_total": line_total,
        })
        orders.setdefault(line["order"], Decimal("0"))
        orders[line["order"]] += line_total

    summary = {}
    for order_id, subtotal in orders.items():
        subtotal = _cents(subtotal)
        tax = _cents(subtotal * TAX_RATE)
        shipping = SHIPPING_FEE if Decimal("0") < subtotal < Decimal("50.00") else _cents(Decimal("0"))
        total = subtotal + tax + shipping
        summary[order_id] = {
            "order": order_id,
            "subtotal": subtotal,
            "tax": tax,
            "shipping": shipping,
            "total": _cents(total),
        }
    return priced, summary
