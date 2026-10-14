"""Entry point: run the orders pipeline and render the invoice text.

    process(orders_csv: str, catalog: dict, stock: dict) -> str

Pipeline: parser.parse_orders -> inventory.allocate -> pricing.price_order.
"""

from decimal import Decimal, ROUND_HALF_UP

from orders import inventory, parser, pricing

CENT = Decimal("0.01")


def _money(amount) -> str:
    return str(Decimal(amount).quantize(CENT, rounding=ROUND_HALF_UP))


def process(orders_csv, catalog, stock):
    lines, parse_rejects = parser.parse_orders(orders_csv)
    allocations, alloc_rejects = inventory.allocate(lines, stock)

    # Group allocated lines by order, preserving input (line_no) order.
    # Only orders with at least one non-rejected line are printed.
    by_order = {}
    for alloc in allocations:
        by_order.setdefault(alloc["order_id"], []).append(alloc)

    blocks = []
    for order_id in sorted(by_order):
        allocs = by_order[order_id]
        priced = pricing.price_order(
            [(a["sku"], a["filled"]) for a in allocs], catalog
        )
        out = [f"ORDER {order_id}"]
        for alloc, line_total in zip(allocs, priced["lines"]):
            name = catalog[alloc["sku"]]["name"]
            unit = _money(catalog[alloc["sku"]]["price"])
            text = (
                f"  {alloc['sku']} {name} x{alloc['filled']} "
                f"@ {unit} = {_money(line_total)}"
            )
            if alloc["backorder"] > 0:
                text += f" (backorder {alloc['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(priced['subtotal'])}")
        out.append(f"  tax {_money(priced['tax'])}")
        out.append(f"  shipping {_money(priced['shipping'])}")
        out.append(f"  total {_money(priced['total'])}")
        blocks.append("\n".join(out))

    rejected = [(r["line_no"], r["reason"]) for r in parse_rejects]
    rejected += [(r["line_no"], r["reason"]) for r in alloc_rejects]
    if rejected:
        out = ["REJECTED"]
        for line_no, reason in sorted(rejected, key=lambda r: r[0]):
            out.append(f"  line {line_no}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
