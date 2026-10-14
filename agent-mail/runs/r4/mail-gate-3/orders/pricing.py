"""Pricing: price lines and orders with exact decimal arithmetic."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
DEFAULT_TAX_RATE = "0.0825"


def _money(value):
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def _unit_price(sku, catalog):
    if sku.startswith("GIFT-"):
        return Decimal("0.00")
    return Decimal(catalog[sku]["price"])


def _line_total(line, catalog):
    unit = _unit_price(line["sku"], catalog)
    total = Decimal(line["filled"]) * unit
    if line["filled"] >= 10:
        total = total * Decimal("0.90")
    return _money(total)


def price(lines, catalog):
    """Return lines in input order with unit price and line total as 2-dp strings."""
    out = []
    for line in lines:
        unit = _money(_unit_price(line["sku"], catalog))
        out.append({
            "n": line["n"],
            "order": line["order"],
            "sku": line["sku"],
            "filled": line["filled"],
            "unit": str(unit),
            "total": str(_line_total(line, catalog)),
        })
    return out


def order_totals(lines, catalog, tax_rate=DEFAULT_TAX_RATE):
    """Return {order: {subtotal, tax, shipping, total}} as 2-dp strings.

    Order keys appear in first-seen order. Uses the filled quantity of each line.
    """
    rate = Decimal(tax_rate)
    subtotals = {}
    for line in lines:
        subtotals.setdefault(line["order"], Decimal("0"))
        subtotals[line["order"]] += _line_total(line, catalog)
    result = {}
    for order, raw in subtotals.items():
        subtotal = _money(raw)
        tax = _money(subtotal * rate)
        shipping = _money(Decimal("5.00")) if Decimal("0") < subtotal < Decimal("50.00") else _money(Decimal("0"))
        total = _money(subtotal + tax + shipping)
        result[order] = {
            "subtotal": str(subtotal),
            "tax": str(tax),
            "shipping": str(shipping),
            "total": str(total),
        }
    return result
