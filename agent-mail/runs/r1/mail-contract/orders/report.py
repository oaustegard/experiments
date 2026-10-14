"""Entry point: run the orders pipeline and render the invoice text."""

from decimal import Decimal

from orders import parser, inventory, pricing


def _money(value) -> str:
    return f"{Decimal(value).quantize(Decimal('0.01')):.2f}"


def process(orders_csv: str, catalog: dict, stock: dict) -> str:
    # 1. parse the order text
    lines, parse_rejected = parser.parse(orders_csv)

    # 2. lines whose sku is missing from the catalog cannot be priced: reject them
    #    before allocation so they do not consume stock
    priceable = [l for l in lines if l["sku"] in catalog]
    catalog_rejected = [(l["n"], "unknown sku") for l in lines if l["sku"] not in catalog]

    # 3. allocate stock to lines
    allocs, alloc_rejected = inventory.allocate(priceable, stock)

    # 4. price the allocated lines (filled 0 prices to 0.00 and still prints), grouped by order
    priced_input = [
        {"n": a["n"], "order": a["order"], "sku": a["sku"], "filled": a["filled"]}
        for a in allocs
    ]
    priced = pricing.price_orders(priced_input, catalog)

    # Index allocations by line number for name/backorder lookup, keep input order.
    alloc_by_n = {a["n"]: a for a in allocs}

    order_ids = sorted(priced.keys())
    blocks = []
    for order_id in order_ids:
        summary = priced[order_id]
        if not summary["lines"]:
            continue
        out = [f"ORDER {order_id}"]
        for pl in summary["lines"]:
            a = alloc_by_n[pl["n"]]
            name = catalog[pl["sku"]]["name"]
            text = (
                f"  {pl['sku']} {name} x{pl['filled']} @ "
                f"{_money(pl['unit_price'])} = {_money(pl['line_total'])}"
            )
            if a["backorder"] > 0:
                text += f" (backorder {a['backorder']})"
            out.append(text)
        out.append(f"  subtotal {_money(summary['subtotal'])}")
        out.append(f"  tax {_money(summary['tax'])}")
        out.append(f"  shipping {_money(summary['shipping'])}")
        out.append(f"  total {_money(summary['total'])}")
        blocks.append("\n".join(out))

    rejected = list(parse_rejected) + catalog_rejected + list(alloc_rejected)
    if rejected:
        rej_lines = ["REJECTED"]
        for n, reason in sorted(rejected, key=lambda r: r[0]):
            rej_lines.append(f"  line {n}: {reason}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
