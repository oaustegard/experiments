"""Entry point: run the order pipeline and render the invoice text.

    process(orders_csv: str, catalog: dict, stock: dict) -> str

Pipeline: parser.parse_lines -> reject lines whose sku is not in catalog ->
inventory.allocate -> pricing.price_order per order -> render.
"""

from orders.parser import parse_lines
from orders.inventory import allocate
from orders.pricing import price_order


def _money(value):
    return f"{value:.2f}"


def process(orders_csv: str, catalog: dict, stock: dict) -> str:
    lines, rejections = parse_lines(orders_csv)

    # Lines whose sku is not in the catalog cannot be priced; reject them here.
    priceable = []
    for ln in lines:
        if ln["sku"] in catalog:
            priceable.append(ln)
        else:
            rejections.append({"line": ln["line"], "reason": "unknown sku"})

    allocs, alloc_rejects = allocate(priceable, stock)
    rejections.extend(alloc_rejects)

    # Group allocated lines by order, keeping file order inside each order.
    by_order = {}
    for alloc in sorted(allocs, key=lambda a: a["line"]):
        by_order.setdefault(alloc["order"], []).append(alloc)

    blocks = []
    for order_id in sorted(by_order):
        group = by_order[order_id]
        priced = price_order(
            [{"sku": a["sku"], "filled": a["filled"]} for a in group], catalog
        )
        out = [f"ORDER {order_id}"]
        for alloc, pl in zip(group, priced["lines"]):
            name = catalog[alloc["sku"]]["name"]
            text = (
                f"  {alloc['sku']} {name} x{alloc['filled']} "
                f"@ {_money(pl['unit'])} = {_money(pl['amount'])}"
            )
            if alloc["backorder"] > 0:
                text += f" (backorder {alloc['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(priced['subtotal'])}")
        out.append(f"  tax {_money(priced['tax'])}")
        out.append(f"  shipping {_money(priced['shipping'])}")
        out.append(f"  total {_money(priced['total'])}")
        blocks.append("\n".join(out))

    if rejections:
        rej_lines = ["REJECTED"]
        for rej in sorted(rejections, key=lambda r: r["line"]):
            rej_lines.append(f"  line {rej['line']}: {rej['reason']}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
