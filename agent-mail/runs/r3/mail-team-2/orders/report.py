"""Entry point: run the order pipeline and render the invoice text."""

from orders.parser import parse
from orders.inventory import allocate
from orders.pricing import price, totals


def _money(d):
    return f"{d:.2f}"


def _block_for_order(order_id, recs, tot):
    out = [f"ORDER {order_id}"]
    has_backorder = False
    for r in recs:
        line = (
            f"  {r['sku']} {r['name']} x{r['filled']} @ {_money(r['unit'])} "
            f"= {_money(r['total'])}"
        )
        if r["backorder"] > 0:
            has_backorder = True
            line += f" (backorder {r['backorder']})"
        out.append(line)
    out.append(f"  subtotal {_money(tot['subtotal'])}")
    out.append(f"  tax {_money(tot['tax'])}")
    out.append(f"  shipping {_money(tot['shipping'])}")
    out.append(f"  total {_money(tot['total'])}")
    out.append("  status PARTIAL" if has_backorder else "  status COMPLETE")
    return "\n".join(out)


def process(orders_csv, catalog, stock):
    lines, rej_parse = parse(orders_csv)
    allocs, rej_alloc = allocate(lines, dict(stock))

    rejected = list(rej_parse) + list(rej_alloc)
    rejected.sort(key=lambda t: t[0])

    priced = price(allocs, catalog)  # same order as allocs

    # Group priced records by order id (input order preserved within group).
    groups = {}
    for a, p in zip(allocs, priced):
        rec = {
            "n": a["n"],
            "sku": a["sku"],
            "name": catalog[a["sku"]]["name"],
            "filled": a["filled"],
            "backorder": a["backorder"],
            "unit": p["unit"],
            "total": p["total"],
        }
        groups.setdefault(a["order"], []).append(rec)

    blocks = []
    for order_id in sorted(groups):
        recs = groups[order_id]
        tot = totals(recs)
        blocks.append(_block_for_order(order_id, recs, tot))

    if rejected:
        rej_lines = ["REJECTED"]
        for n, reason in rejected:
            rej_lines.append(f"  line {n}: {reason}")
        blocks.append("\n".join(rej_lines))

    if not blocks:
        return "NO ORDERS"
    return "\n\n".join(blocks)
