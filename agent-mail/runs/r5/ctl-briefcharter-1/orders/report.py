from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price, order_totals


def process(orders_csv, catalog, stock):
    lines, rejected = parse(orders_csv)
    allocs, alloc_rejected = allocate(lines, stock)
    rejected = sorted(list(rejected) + list(alloc_rejected), key=lambda r: r[0])

    priced = price(allocs, catalog)
    totals = order_totals(priced)

    by_order = {}
    for line in priced:
        by_order.setdefault(line["order"], []).append(line)

    blocks = []
    for order_id in sorted(by_order):
        order_lines = by_order[order_id]
        out = [f"ORDER {order_id}"]
        for line in order_lines:
            name = catalog[line["sku"]]["name"]
            text = (f"  {line['sku']} {name} x{line['filled']}"
                    f" @ {line['unit']} = {line['line_total']}")
            if line["backorder"] > 0:
                text += f" (backorder {line['backorder']})"
            out.append(text)
        t = totals[order_id]
        out.append(f"  subtotal {t['subtotal']}")
        out.append(f"  tax {t['tax']}")
        out.append(f"  shipping {t['shipping']}")
        out.append(f"  total {t['total']}")
        partial = any(line["backorder"] > 0 for line in order_lines)
        out.append("  status PARTIAL" if partial else "  status COMPLETE")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in rejected:
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
