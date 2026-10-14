"""Entry point: run the order pipeline and render the invoice text.

process(orders_csv, catalog, stock) -> str
"""

from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price

CENT = Decimal("0.01")


def _money(value):
    return str(Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP))


def _order_block(order_id, lines, summary):
    out = [f"ORDER {order_id}"]
    partial = False
    for line in lines:
        name = line["name"]
        text = (
            f"  {line['sku']} {name} x{line['filled']} "
            f"@ {_money(line['unit'])} = {_money(line['line_total'])}"
        )
        if line["backorder"] > 0:
            text += f" (backorder {line['backorder']})"
            partial = True
        out.append(text)
    s = summary[order_id]
    out.append(f"  subtotal {_money(s['subtotal'])}")
    out.append(f"  tax {_money(s['tax'])}")
    out.append(f"  shipping {_money(s['shipping'])}")
    out.append(f"  total {_money(s['total'])}")
    out.append(f"  status {'PARTIAL' if partial else 'COMPLETE'}")
    return "\n".join(out)


def process(orders_csv, catalog, stock):
    # Parser uppercases SKUs, so normalize catalog/stock keys to match.
    stock = {str(k).upper(): v for k, v in stock.items()}
    catalog = {str(k).upper(): v for k, v in catalog.items()}

    lines, parse_rejected = parse(orders_csv)
    allocs, alloc_rejected = allocate(lines, stock)

    rejected = list(parse_rejected) + list(alloc_rejected)
    rejected.sort(key=lambda r: r[0])

    # Price only lines that were not rejected; pricing needs `filled`.
    priced, summary = price(allocs, catalog)

    # Attach name / backorder / unit / line_total per allocated line, keyed by n.
    priced_by_n = {p["n"]: p for p in priced}
    by_order = {}
    for alloc in allocs:
        p = priced_by_n[alloc["n"]]
        entry = {
            "n": alloc["n"],
            "sku": alloc["sku"],
            "filled": alloc["filled"],
            "backorder": alloc["backorder"],
            "unit": p["unit"],
            "line_total": p["line_total"],
            "name": catalog.get(alloc["sku"], {}).get("name", alloc["sku"]),
        }
        by_order.setdefault(alloc["order"], []).append(entry)

    blocks = []
    for order_id in sorted(by_order):
        blocks.append(_order_block(order_id, by_order[order_id], summary))

    if rejected:
        rej_lines = ["REJECTED"]
        for n, reason in rejected:
            rej_lines.append(f"  line {n}: {reason}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
