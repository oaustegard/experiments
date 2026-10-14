import json, random, sys
sys.path.insert(0, sys.argv[1]); from ref import process
R = random.Random(7)
CAT = {"WIDGET": {"name": "Blue Widget", "price": "2.50"}, "GADGET": {"name": "Gadget", "price": "10.00"},
       "BOLT": {"name": "Hex Bolt", "price": "0.35"}, "LAMP": {"name": "Desk Lamp", "price": "19.99"},
       "CABLE": {"name": "USB Cable", "price": "4.45"}, "GHOST": {"name": "Ghost", "price": "1.00"}}
def case():
    stock = {k: R.choice([0, 3, 8, 12, 40, 200]) for k in CAT if k != "GHOST"}
    out = []
    if R.random() < .3: out += [R.choice(["# orders export", ""])]
    if R.random() < .6: out.append(R.choice(["order_id,sku,qty", "Order_ID, SKU, Qty", "ORDER_ID,SKU,QTY"]))
    for _ in range(R.randint(0, 9)):
        o = R.choice(["A1", "A10", "A2", "B7", "b7", "C3"]); s = R.choice(list(CAT) + ["widget", " lamp ", "NOPE"])
        q = R.choice(["1", "3", "5", "10", "12", " 4 ", "0", "-2", "x", "2.5", "25"])
        r = R.random()
        if r < .06: out.append(f"{o},{s}")
        elif r < .1: out.append(f",{s},{q}")
        elif r < .14: out.append(f"{o},{s},{q},extra")
        elif r < .2: out.append(R.choice(["", "  # note", "   "]))
        else: out.append(f"{o},{s},{q}")
    return "\n".join(out), CAT, stock
cases = []
fixed = [("order_id,sku,qty\nA1,WIDGET,3\nA1,GADGET,5", CAT, {"WIDGET": 10, "GADGET": 3}), ("", CAT, {}), ("# only\n\n", CAT, {"WIDGET": 1})]
for t, c, s in fixed + [case() for _ in range(45)]:
    cases.append({"text": t, "catalog": c, "stock": s, "expect": process(t, c, s)})
json.dump(cases, open(sys.argv[2], "w"), indent=1)
print(len(cases)); print(cases[0]["expect"]); print("---"); print(cases[5]["text"]); print(cases[5]["expect"])
