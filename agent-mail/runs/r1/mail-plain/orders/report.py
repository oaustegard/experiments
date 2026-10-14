"""Entry point: run the order pipeline and render the invoice text.

Public interface:

    process(orders_csv: str, catalog: dict, stock: dict) -> str

Pipeline:
    parser.parse(orders_csv)                -> (lines, rejects)
    inventory.allocate(lines, stock)        -> (allocations, rejects)
    pricing.price(allocations, catalog)     -> priced lines (name, unit_price, line_total)
    pricing.order_totals(priced_lines)      -> {subtotal, tax, shipping, total}
"""

from decimal import Decimal, ROUND_HALF_UP

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price, order_totals


NO_ORDERS = "NO ORDERS"


def _money(value) -> str:
    return f"{Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def _render_order(order_id: str, priced_lines: list[dict]) -> str:
    totals = order_totals(priced_lines)
    out = [f"ORDER {order_id}"]
    for ln in priced_lines:
        text = (
            f"  {ln['sku']} {ln['name']} x{ln['filled']}"
            f" @ {_money(ln['unit_price'])} = {_money(ln['line_total'])}"
        )
        if ln.get("backorder", 0) > 0:
            text += f" (backorder {ln['backorder']})"
        out.append(text)
    out.append(f"  subtotal {_money(totals['subtotal'])}")
    out.append(f"  tax {_money(totals['tax'])}")
    out.append(f"  shipping {_money(totals['shipping'])}")
    out.append(f"  total {_money(totals['total'])}")
    return "\n".join(out)


def _render_rejects(rejects: list[dict]) -> str:
    out = ["REJECTED"]
    for rj in sorted(rejects, key=lambda r: r["line"]):
        out.append(f"  line {rj['line']}: {rj['reason']}")
    return "\n".join(out)


def process(orders_csv: str, catalog: dict, stock: dict) -> str:
    lines, parse_rejects = parse(orders_csv)
    allocations, alloc_rejects = allocate(lines, stock)
    priced = price(allocations, catalog)

    rejects = list(parse_rejects) + list(alloc_rejects)

    # Group priced lines by order, keeping input line order within each order.
    by_order: dict[str, list[dict]] = {}
    for ln in sorted(priced, key=lambda r: r["line"]):
        by_order.setdefault(ln["order_id"], []).append(ln)

    blocks = [_render_order(oid, by_order[oid]) for oid in sorted(by_order)]
    if rejects:
        blocks.append(_render_rejects(rejects))

    if not blocks:
        return NO_ORDERS
    return "\n\n".join(blocks)
