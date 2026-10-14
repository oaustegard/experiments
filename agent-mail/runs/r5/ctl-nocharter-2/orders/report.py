"""Entry point: run the order pipeline and render the invoice text."""
from decimal import Decimal

from orders.inventory import allocate
from orders.parser import parse
from orders.pricing import price_orders


def process(orders_csv, catalog, stock):
    lines, rejected = parse(orders_csv)

    accepted = []
    for line in lines:
        if line["sku"] in catalog:
            accepted.append(line)
        else:
            rejected.append((line["n"], "unknown sku"))

    # Catalog skus missing from stock have zero units on hand.
    full_stock = {sku: stock.get(sku, 0) for sku in catalog}
    allocs, alloc_rejected = allocate(accepted, full_stock)
    rejected.extend(alloc_rejected)

    invoices = {inv["order"]: inv for inv in price_orders(allocs, catalog)}

    by_order = {}
    for alloc in allocs:
        by_order.setdefault(alloc["order"], []).append(alloc)

    blocks = []
    for order in sorted(by_order):
        invoice = invoices[order]
        line_totals = {line["sku"]: line["total"] for line in invoice["lines"]}
        out = [f"ORDER {order}"]
        for alloc in by_order[order]:
            sku = alloc["sku"]
            name = catalog[sku]["name"]
            unit = Decimal(catalog[sku]["price"])
            total = line_totals.get(sku, Decimal("0.00"))
            text = f"  {sku} {name} x{alloc['filled']} @ {unit:.2f} = {total:.2f}"
            if alloc["backorder"] > 0:
                text += f" (backorder {alloc['backorder']})"
            out.append(text)
        out.append(f"  subtotal {invoice['subtotal']:.2f}")
        out.append(f"  tax {invoice['tax']:.2f}")
        out.append(f"  shipping {invoice['shipping']:.2f}")
        out.append(f"  total {invoice['total']:.2f}")
        blocks.append("\n".join(out))

    if rejected:
        out = ["REJECTED"]
        for n, reason in sorted(rejected):
            out.append(f"  line {n}: {reason}")
        blocks.append("\n".join(out))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
