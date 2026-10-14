"""Entry point: run the order pipeline and render the invoice text."""
from orders import inventory, parser, pricing


def process(orders_csv, catalog, stock):
    lines, rejected = parser.parse(orders_csv)

    known = []
    for line in lines:
        if line["sku"] in catalog:
            known.append(line)
        else:
            rejected.append((line["n"], "unknown sku"))

    allocated, alloc_rejected = inventory.allocate(known, stock)
    rejected.extend(alloc_rejected)

    priced = pricing.price_lines(allocated, catalog)

    by_order = {}
    for line, p in zip(allocated, priced):
        by_order.setdefault(line["order"], []).append((line, p))

    blocks = []
    for order_id in sorted(by_order):
        pairs = by_order[order_id]
        out = [f"ORDER {order_id}"]
        for line, p in pairs:
            name = catalog[line["sku"]]["name"]
            row = (f"  {line['sku']} {name} x{line['filled']} @ "
                   f"{p['unit']} = {p['total']}")
            if line["backorder"] > 0:
                row += f" (backorder {line['backorder']})"
            out.append(row)
        totals = pricing.summarize([p for _, p in pairs])
        out.append(f"  subtotal {totals['subtotal']}")
        out.append(f"  tax {totals['tax']}")
        out.append(f"  shipping {totals['shipping']}")
        out.append(f"  total {totals['total']}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in sorted(rejected):
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
