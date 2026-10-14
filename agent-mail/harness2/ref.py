from decimal import Decimal, ROUND_HALF_UP
C = Decimal("0.01")
def r(x): return x.quantize(C, rounding=ROUND_HALF_UP)

RULES = frozenset('ABCD')  # A gift free, B reserve 2, C status line, D order_id upper

def process(text, catalog, stock, rules=None):
    rules = RULES if rules is None else rules
    rej, lines, idx, header_done = [], [], {}, False
    for n, raw in enumerate(text.split("\n"), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if not header_done:
            header_done = True
            if s.replace(" ", "").lower() == "order_id,sku,qty":
                continue
        f = [x.strip() for x in raw.split(",")]
        if len(f) != 3 or not f[0] or not f[1]:
            rej.append((n, "bad format")); continue
        try:
            q = int(f[2])
        except ValueError:
            q = 0
        if q <= 0:
            rej.append((n, "bad qty")); continue
        oid = f[0].upper() if "D" in rules else f[0]
        k = (oid, f[1].upper())
        if k in idx:
            idx[k]["qty"] += q
        else:
            idx[k] = {"n": n, "order": oid, "sku": f[1].upper(), "qty": q}
            lines.append(idx[k])
    left = {k: (max(v - 2, 0) if "B" in rules else v) for k, v in stock.items()}
    alloc = []
    for L in sorted(lines, key=lambda l: l["n"]):
        if L["sku"] not in left:
            rej.append((L["n"], "unknown sku")); continue
        fill = min(L["qty"], left[L["sku"]]); left[L["sku"]] -= fill
        alloc.append(dict(L, filled=fill, back=L["qty"] - fill))
    blocks = []
    for oid in sorted({a["order"] for a in alloc}):
        out, sub = [f"ORDER {oid}"], Decimal(0)
        for a in alloc:
            if a["order"] != oid: continue
            unit = Decimal("0") if "A" in rules and a["sku"].startswith("GIFT-") else Decimal(catalog[a["sku"]]["price"])
            t = a["filled"] * unit
            if a["filled"] >= 10: t = t * Decimal("0.90")
            t = r(t); sub += t
            s = f"  {a['sku']} {catalog[a['sku']]['name']} x{a['filled']} @ {r(unit)} = {t}"
            if a["back"]: s += f" (backorder {a['back']})"
            out.append(s)
        tax = r(sub * Decimal("0.0825"))
        ship = Decimal("5.00") if 0 < sub < 50 else Decimal("0.00")
        out += [f"  subtotal {r(sub)}", f"  tax {tax}", f"  shipping {ship}", f"  total {r(sub + tax + ship)}"]
        if "C" in rules: out.append("  status " + ("PARTIAL" if any(a["back"] for a in alloc if a["order"] == oid) else "COMPLETE"))
        blocks.append("\n".join(out))
    if rej:
        blocks.append("\n".join(["REJECTED"] + [f"  line {n}: {why}" for n, why in sorted(rej)]))
    return "\n\n".join(blocks) if blocks else "NO ORDERS"
