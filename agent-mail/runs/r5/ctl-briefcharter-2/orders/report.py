"""Entry point: run the order pipeline and render the invoice text."""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price_orders


def process(orders_csv, catalog, stock):
    """Return the invoice text for the given order CSV, catalog and stock."""
    lines, rejected = parse(orders_csv)

    known = []
    for line in lines:
        if line["sku"] in catalog:
            known.append(line)
        else:
            rejected.append((line["n"], "unknown sku"))

    allocs, alloc_rejected = allocate(known, stock)
    rejected = list(rejected) + list(alloc_rejected)

    filled_lines = [
        {"n": a["n"], "order": a["order"], "sku": a["sku"], "filled": a["filled"]}
        for a in allocs
    ]
    backorder = {a["n"]: a["backorder"] for a in allocs}

    blocks = []
    for order in sorted(price_orders(filled_lines, catalog), key=lambda o: o["order"]):
        out = ["ORDER " + order["order"]]
        has_backorder = False
        for line in order["lines"]:
            sku = line["sku"]
            name = catalog[sku]["name"]
            text = "  %s %s x%d @ %s = %s" % (
                sku, name, line["filled"], line["unit"], line["total"],
            )
            if backorder[line["n"]] > 0:
                text += " (backorder %d)" % backorder[line["n"]]
                has_backorder = True
            out.append(text)
        out.append("  subtotal " + order["subtotal"])
        out.append("  tax " + order["tax"])
        out.append("  shipping " + order["shipping"])
        out.append("  total " + order["total"])
        out.append("  status " + ("PARTIAL" if has_backorder else "COMPLETE"))
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in sorted(rejected, key=lambda r: r[0]):
            out.append("  line %d: %s" % (n, reason))
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
