"""Everyday spikes (system v0.2.0): data for three domains anyone can judge.

    python3 -I run/everyday_data.py transactions          # generated, labels known by construction
    python3 -I run/everyday_data.py reviews                # Yelp reviews (stars from the data), then label
    python3 -I run/everyday_data.py emails                 # Enron ham/spam + Sonnet-written newsletters, bills,
                                                           #   promotions and personal mail, then label
    python3 -I run/everyday_data.py label reviews|emails   # Sonnet (claude -p) answers the typed questions
    python3 -I run/everyday_data.py build                  # -> data/everyday/cases.jsonl (train/test split)

Every case has the Typed Decisions shape (id, workflow, state, questions, gold, split) so lora.py and
the page read it like the benchmark. Gold is one-hot. Sonnet labels go to data/everyday/labels_<d>.jsonl,
one line per item, resume-safe.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "everyday"
N_PER_DOMAIN, TEST_FRAC = 1500, 0.2

QUESTIONS = {
    "transactions": {
        "category": {"type": "choice", "instructions": "What kind of spending or income is this bank transaction?", "criteria": {
            "groceries": "Supermarkets and grocery stores.", "dining": "Restaurants, cafés, bars, takeaway and food delivery.",
            "transport": "Fuel, rideshare, transit, parking and tolls.", "shopping": "General retail and online shopping.",
            "household": "Home goods, hardware, furniture and household supplies.",
            "subscriptions": "Streaming, software, memberships and other subscriptions.",
            "utilities": "Electricity, gas, water, internet and phone bills.", "entertainment": "Cinema, concerts, events, games and hobbies.",
            "travel": "Flights, hotels, car rental and other travel bookings.", "health": "Pharmacy, doctors, dentists and gyms.",
            "income": "Salary, refunds and other money coming in.", "transfer": "Payments to people and moves between accounts."}},
        "recurring": {"type": "noul", "instructions": "This is a recurring charge: a subscription, a bill or another regular payment.",
                      "criteria": {"true": "It repeats on a regular schedule.", "false": "It is a one-off or irregular transaction."}},
        "unusual": {"type": "noul", "instructions": "This transaction is unusual: much larger than normal for this merchant or out of its usual pattern.",
                    "criteria": {"true": "It stands out and is worth a second look.", "false": "It looks normal for this merchant."}},
    },
    "reviews": {
        "stars": {"type": "score", "instructions": "How many stars does this review give?", "criteria": [
            "1 star: very negative.", "2 stars: negative.", "3 stars: mixed.", "4 stars: positive.", "5 stars: very positive."]},
        "complaint": {"type": "choice", "instructions": "What is the reviewer's main complaint, if any?", "criteria": {
            "none": "No real complaint.", "quality": "The food or product itself was poor.", "service": "Staff were rude, unhelpful or careless.",
            "price": "Too expensive or poor value.", "wait": "Slow service, long waits or delays.", "cleanliness": "Dirty or unhygienic.",
            "other": "Some other complaint."}},
        "needs_reply": {"type": "noul", "instructions": "The business should publicly reply to this review.",
                        "criteria": {"true": "A reply would help: a complaint to address, a question, or a factual error.",
                                     "false": "No reply is needed."}},
    },
    "emails": {
        "category": {"type": "choice", "instructions": "What kind of email is this?", "criteria": {
            "personal": "From friends or family, or about the recipient's private life.", "work": "About the recipient's job or business.",
            "bill": "A bill, receipt, invoice, statement or payment notice.", "newsletter": "A newsletter or update the recipient subscribed to.",
            "promotion": "A marketing offer or sale from a company.", "spam": "Unwanted bulk mail, a scam or phishing."}},
        "needs_reply": {"type": "noul", "instructions": "The recipient needs to reply to this email.",
                        "criteria": {"true": "Someone is waiting on an answer from the recipient.", "false": "No reply is needed."}},
        "urgency": {"type": "score", "instructions": "How soon does the recipient need to act on this email?", "criteria": [
            "No action or no time pressure.", "Within a week or so.", "Within a day or two.", "Today or immediately."]},
    },
}

# ---- transactions: a merchant table; labels follow from how each transaction is generated ----
M = [  # (descriptor template, category, typical amount range, recurring?)
    ("WHOLEFDS #{n} {city}", "groceries", (30, 160), False), ("TRADER JOE'S #{n}", "groceries", (20, 120), False),
    ("SAFEWAY {n}", "groceries", (15, 140), False), ("ALDI {n} {city}", "groceries", (15, 90), False),
    ("SQ *BLUE BOTTLE COFFEE", "dining", (4, 14), False), ("STARBUCKS STORE {n}", "dining", (3, 15), False),
    ("DOORDASH*{shop}", "dining", (15, 60), False), ("UBER *EATS {code}", "dining", (15, 55), False),
    ("TST* {shop} {city}", "dining", (20, 120), False), ("CHIPOTLE {n}", "dining", (9, 25), False),
    ("SHELL OIL {n}", "transport", (25, 80), False), ("UBER *TRIP {code}", "transport", (8, 45), False),
    ("LYFT *RIDE {code}", "transport", (8, 45), False), ("CLIPPER TRANSIT", "transport", (20, 60), False),
    ("PARKMOBILE {code}", "transport", (2, 18), False),
    ("AMZN MKTP US*{code}", "shopping", (8, 150), False), ("TARGET {n}", "shopping", (10, 150), False),
    ("ETSY.COM*{shop}", "shopping", (10, 90), False), ("BEST BUY {n}", "shopping", (20, 400), False),
    ("HOME DEPOT #{n}", "household", (10, 250), False), ("IKEA {city}", "household", (25, 400), False),
    ("WAYFAIR*{code}", "household", (30, 500), False), ("ACE HARDWARE {n}", "household", (5, 60), False),
    ("NETFLIX.COM", "subscriptions", (15.49, 15.49), True), ("SPOTIFY USA", "subscriptions", (11.99, 11.99), True),
    ("APPLE.COM/BILL", "subscriptions", (2.99, 2.99), True), ("ADOBE *CREATIVE CLD", "subscriptions", (59.99, 59.99), True),
    ("NYTIMES*DIGITAL", "subscriptions", (17.0, 17.0), True), ("COSTCO MEMBERSHIP", "subscriptions", (65.0, 65.0), True),
    ("PG&E WEB ONLINE", "utilities", (60, 220), True), ("COMCAST XFINITY", "utilities", (79.99, 79.99), True),
    ("VERIZON WIRELESS", "utilities", (85.0, 85.0), True), ("CITY WATER DEPT", "utilities", (30, 90), True),
    ("AMC THEATRES {n}", "entertainment", (12, 45), False), ("TICKETMASTER*{code}", "entertainment", (40, 300), False),
    ("STEAMGAMES.COM {code}", "entertainment", (5, 60), False), ("EVENTBRITE*{shop}", "entertainment", (10, 90), False),
    ("UNITED AIR {code}", "travel", (120, 700), False), ("DELTA AIR {code}", "travel", (120, 700), False),
    ("MARRIOTT {city}", "travel", (120, 450), False), ("AIRBNB *{code}", "travel", (90, 600), False),
    ("HERTZ RENT-A-CAR", "travel", (60, 350), False),
    ("CVS/PHARMACY #{n}", "health", (5, 60), False), ("WALGREENS #{n}", "health", (5, 50), False),
    ("PLANET FITNESS", "health", (24.99, 24.99), True), ("BAY DENTAL GROUP", "health", (80, 300), False),
    ("ACME CORP PAYROLL", "income", (2400, 2400), True), ("IRS TREAS 310 TAX REF", "income", (300, 1500), False),
    ("AMAZON REFUND {code}", "income", (10, 120), False),
    ("ZELLE TO {person}", "transfer", (20, 400), False), ("VENMO *{person}", "transfer", (10, 150), False),
    ("TRANSFER TO SAVINGS", "transfer", (100, 500), True),
]
CITIES = ["OAKLAND CA", "SEATTLE WA", "AUSTIN TX", "BROOKLYN NY", "DENVER CO"]
SHOPS = ["MAMA LUCIAS", "THAI HOUSE", "GREENLEAF", "CORNER BISTRO", "NORTHWIND", "LITTLE ITALY"]
PEOPLE = ["J SMITH", "MARIA G", "ALEX K", "SAM P", "PRIYA R"]


def transaction(i, rng):
    tpl, cat, (lo, hi), rec = rng.choice(M)
    desc = tpl.format(n=rng.randint(100, 9999), city=rng.choice(CITIES), shop=rng.choice(SHOPS),
                      code=f"{rng.choice('ABCDEFGHJKMNPQRSTUVWXYZ')}{rng.randint(10, 99)}{rng.choice('ABCDEFGHJKMNPQRSTUVWXYZ')}{rng.randint(1, 9)}",
                      person=rng.choice(PEOPLE))
    typical = round(rng.uniform(lo, hi), 2)
    unusual = cat not in ("income", "transfer") and rng.random() < 0.15
    amount = round(typical * (rng.uniform(4, 9) if unusual else 1.0), 2)
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    hist = []
    if rec:
        hist = [{"date": f"2026-{(month - k - 2) % 12 + 1:02d}-{day:02d}", "amount_usd": typical} for k in range(rng.randint(2, 5))]
    elif rng.random() < 0.4:
        hist = [{"date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", "amount_usd": round(rng.uniform(lo, hi), 2)}
                for _ in range(rng.randint(1, 3))]
    sign = 1 if cat == "income" else -1
    state = {"transaction": {"description": desc, "amount_usd": sign * amount, "date": f"2026-{month:02d}-{day:02d}",
                             "account": rng.choice(["checking", "credit card"])},
             "history_same_merchant": [{**h, "amount_usd": sign * h["amount_usd"]} for h in hist]}
    labels = {"category": cat, "recurring": "true" if rec else "false", "unusual": "true" if unusual else "false"}
    return {"id": f"tx_{i:05d}", "state": json.dumps(state), "labels": labels}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def claude(system, prompt, timeout=900):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["claude", "-p", "--model", "sonnet", "--output-format", "json", "--setting-sources", "", "--strict-mcp-config",
               "--no-session-persistence", "--tools", "", "--system-prompt", system, prompt]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, timeout=timeout)
    env = json.loads(r.stdout)
    return env.get("result", ""), env.get("total_cost_usd") or 0.0


def reviews():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("Yelp/yelp_review_full", "yelp_review_full/train-00000-of-00001.parquet", repo_type="dataset"))
    df = df[df.text.str.len().between(80, 1200)]
    rows = []
    for star in range(5):   # balanced across 1-5 stars
        for k, r in enumerate(df[df.label == star].sample(N_PER_DOMAIN // 5, random_state=star).itertuples()):
            rows.append({"id": f"rv_{star}{k:04d}", "state": json.dumps({"review": r.text.replace("\\n", "\n")}),
                         "labels": {"stars": str(star)}})
    write_jsonl(D / "items_reviews.jsonl", rows)
    print(f"reviews: {len(rows)}")


GEN = {"newsletter": "newsletters people subscribe to (a local food blog digest, a running club update, a software release note, a museum's monthly events, a neighbourhood association bulletin)",
       "promotion": "marketing emails from shops and services (sales, discount codes, new arrivals, loyalty points)",
       "bill": "bills, receipts, statements and payment notices (utility bills, order receipts, card statements, rent reminders, overdue notices)",
       "personal": "personal emails between friends and family (plans, news, favours, condolences, invitations), some needing a reply and some not"}


def emails():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = []
    for split in ("train",):
        df = pd.read_json(hf_hub_download("SetFit/enron_spam", f"{split}.jsonl", repo_type="dataset"), lines=True)
        df = df[df.message.str.len().between(60, 1500)]
        for k, r in enumerate(df[df.label == 0].sample(700, random_state=1).itertuples()):
            rows.append({"id": f"em_h{k:04d}", "state": json.dumps({"subject": r.subject, "body": r.message}), "labels": {}})
        for k, r in enumerate(df[df.label == 1].sample(200, random_state=2).itertuples()):
            rows.append({"id": f"em_s{k:04d}", "state": json.dumps({"subject": r.subject, "body": r.message}), "labels": {}})
    system = ("You write realistic sample emails for a classifier's training data. Use fictional people and fictional companies "
              "only; never real brands. Vary length, tone and detail. Reply with only a JSON list of objects "
              '{"subject": ..., "body": ...}.')
    jobs = [(t, j) for t in GEN for j in range(8)]          # 8 calls x 20 emails per type

    def gen(job):
        t, j = job
        text, cost = claude(system, f"Write 20 different emails of this kind: {GEN[t]}. Batch {j}: make them unlike other batches.")
        m = re.search(r"\[[\s\S]*\]", text)
        return t, j, (json.loads(m.group(0)) if m else []), cost

    total = 0.0
    with ThreadPoolExecutor(6) as ex:
        for t, j, items, cost in ex.map(gen, jobs):
            total += cost
            for k, it in enumerate(items[:20]):
                rows.append({"id": f"em_g{t[:3]}{j}{k:02d}", "state": json.dumps({"subject": it["subject"], "body": it["body"]}),
                             "labels": {}, "source": f"sonnet-{t}"})
    write_jsonl(D / "items_emails.jsonl", rows)
    print(f"emails: {len(rows)} (generation ${total:.2f})")


def label(domain, batch=20, workers=6):
    items = read_jsonl(D / f"items_{domain}.jsonl")
    out_path = D / f"labels_{domain}.jsonl"
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    qs = {q: v for q, v in QUESTIONS[domain].items() if not (domain == "reviews" and q == "stars")}

    def opts(q):
        c = q["criteria"]
        return {str(i): d for i, d in enumerate(c)} if isinstance(c, list) else c
    spec = "\n".join(f"- {name} ({q['type']}): {q['instructions']} Options: " + "; ".join(f"{o} = {d}" for o, d in opts(q).items())
                     for name, q in qs.items())
    system = ("You label items for a classifier. For each item, answer every question with exactly one option key. "
              "Judge from the item alone, as a careful person would. Reply with only a JSON object mapping each item id to "
              "an object of question -> option key.\n\nQuestions:\n" + spec)
    batches = [todo[i: i + batch] for i in range(0, len(todo), batch)]

    def run(b):
        prompt = "\n\n".join(f"=== ITEM {it['id']} ===\n{it['state']}" for it in b)
        for attempt in range(2):
            try:
                text, cost = claude(system, prompt)
                rep = json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
                good = []
                for it in b:
                    a = rep.get(it["id"], {})
                    if all(str(a.get(q)) in opts(v) for q, v in qs.items()):
                        good.append({"id": it["id"], "labels": {q: str(a[q]) for q in qs}})
                return good, cost
            except Exception as e:  # noqa: BLE001 - one retry, then the batch is left for a rerun
                err = e
        print("batch failed:", str(err)[:120], flush=True)
        return [], 0.0

    total, n = 0.0, 0
    with ThreadPoolExecutor(workers) as ex, open(out_path, "a") as f:
        for good, cost in ex.map(run, batches):
            total += cost
            n += len(good)
            for g in good:
                f.write(json.dumps(g) + "\n")
            f.flush()
            print(f"{domain}: +{len(good)} labelled, ${total:.2f}", flush=True)
    print(f"{domain}: {n} new labels, {len(todo) - n} unlabelled, ${total:.2f}")


def build():
    rows = []
    rng = random.Random(0)
    for i in range(N_PER_DOMAIN):
        rows.append({**transaction(i, rng), "workflow": "transactions"})
    for domain in ("reviews", "emails"):
        lab = {r["id"]: r["labels"] for r in read_jsonl(D / f"labels_{domain}.jsonl")}
        for it in read_jsonl(D / f"items_{domain}.jsonl"):
            if it["id"] in lab:
                rows.append({"id": it["id"], "workflow": domain, "state": it["state"], "labels": {**it["labels"], **lab[it["id"]]}})
    out = []
    for r in rows:
        qs = QUESTIONS[r["workflow"]]
        gold = {}
        for q, spec in qs.items():
            c = spec["criteria"]
            opts = [str(i) for i in range(len(c))] if isinstance(c, list) else list(c)
            gold[q] = {"label": r["labels"][q], "probabilities": {o: float(o == r["labels"][q]) for o in opts}, "type": spec["type"]}
        h = int(re.sub(r"\D", "", r["id"]) or 0) * 2654435761 % 1000
        out.append({"id": r["id"], "workflow": r["workflow"], "state": r["state"], "questions": qs, "gold": gold,
                    "split": "test" if random.Random(r["id"]).random() < TEST_FRAC else "train"})
    write_jsonl(D / "cases.jsonl", out)
    from collections import Counter
    print(Counter((r["workflow"], r["split"]) for r in out))


OFFTOPIC_TRAIN = ["a recipe with ingredients and steps", "a short poem", "a paragraph from a news article", "a social media post",
                  "a paragraph from a novel", "a set of meeting notes as bullet points"]
OFFTOPIC_TEST = ["a code snippet with a comment", "a sports match result summary", "a weather forecast", "song lyrics (original)",
                 "a product specification sheet"]


def offtopic():
    """Texts that belong to none of the three spikes; train and test use different kinds."""
    system = ("You write varied sample texts for a classifier's training data. Fictional names and places only. "
              'Reply with only a JSON list of strings.')
    jobs = [(k, "train") for k in OFFTOPIC_TRAIN] + [(k, "test") for k in OFFTOPIC_TEST]

    def gen(job):
        kind, split = job
        text, cost = claude(system, f"Write 15 different examples of {kind}, each 1-8 sentences or lines.")
        m = re.search(r"\[[\s\S]*\]", text)
        return kind, split, (json.loads(m.group(0)) if m else []), cost

    rows, total = [], 0.0
    with ThreadPoolExecutor(6) as ex:
        for kind, split, items, cost in ex.map(gen, jobs):
            total += cost
            rows += [{"kind": kind, "split": split, "text": t} for t in items[:15] if isinstance(t, str)]
    write_jsonl(D / "offtopic.jsonl", rows)
    print(f"offtopic: {len(rows)} (${total:.2f})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "transactions":
        print(json.dumps(transaction(0, random.Random(0)), indent=1))
    elif cmd == "reviews":
        reviews()
    elif cmd == "emails":
        emails()
    elif cmd == "label":
        label(sys.argv[2])
    elif cmd == "offtopic":
        offtopic()
    elif cmd == "build":
        build()
