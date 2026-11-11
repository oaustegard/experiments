"""Write run/fixtures/jetstream_events.json: about 200 fictional Jetstream events for run/page_smoke2.mjs.

Two phases. In phase 1 every post that passes the default firehose filters (lang en, no replies, at least
20 characters) mentions Microsoft; in phase 2 none does, so with a small window Microsoft has to leave the
grouped view. Mixed in throughout: identity and account events, post deletes and updates, a like commit,
replies, posts in other languages or with no language, short and empty posts, and "giveaway" posts for an
exclude filter. The first post that passes carries <img onerror> and <script> markup. Deterministic.

    python3 run/make_jetstream_fixture.py
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "jetstream_events.json"
R = random.Random(20261111)
B32 = "abcdefghijklmnopqrstuvwxyz234567"
T0 = 1762862400_000000  # 2025-11-11T12:00:00Z in microseconds
DIDS = ["did:plc:" + "".join(R.choice(B32) for _ in range(24)) for _ in range(30)]
CITIES = ["Lisbon", "Oslo", "Dublin", "Madrid", "Austin", "Nairobi"]
PEOPLE = ["Kara Swisher", "Casey Newton", "Ana Silva", "Lars Berg"]

P1 = [
    "Microsoft just announced a new data centre in {city}. Big week for the city.",
    "Satya Nadella said Microsoft will hire 500 engineers in {city} next year.",
    "Watched the Microsoft keynote with {person}; Satya Nadella spent an hour on Copilot.",
    "Microsoft and OpenAI are renegotiating their deal again, according to {person}.",
    "Our team in {city} finally moved everything off Microsoft Teams. Nobody misses it.",
]
P2 = [
    "Tim Cook says Apple will open a second store in {city} this spring.",
    "Sundar Pichai showed the new Google phones in {city} today, {person} was there.",
    "Jensen Huang told investors that Nvidia sold out of chips for the whole year.",
    "{person} argues Apple and Google both depend on the search deal more than they admit.",
    "Spent the morning with {person} talking about why Nvidia is now bigger than Apple.",
]
XSS = 'Satya Nadella says Microsoft is fine <img src=x onerror="window.__xss=1"> <script>window.__xss=2</script> really.'
FOREIGN = [
    (["pt"], "A Microsoft abriu um novo escritório em Lisboa esta semana."),
    (["de"], "Tim Cook hat heute in Berlin über Apple gesprochen."),
    (["ja"], "今日はAppleの新しいiPhoneを東京で見ました。とても良かった。"),
    (["es"], "Sundar Pichai presentó los nuevos teléfonos de Google en Madrid."),
    (["pt", "es"], "Jensen Huang esteve em São Paulo para falar da Nvidia."),
]
SHORT = ["lol", "Apple!", "gm everyone", "so true", "Microsoft :(", "   ok   "]
GIVEAWAY = "GIVEAWAY: win a new Apple iPhone from {person}, repost to enter by Friday!"

events = []


def tid():
    return "3m" + "".join(R.choice(B32) for _ in range(11))


def post(text, langs=("en",), reply=False):
    rec = {"$type": "app.bsky.feed.post", "text": text, "createdAt": "2025-11-11T12:00:00.000Z"}
    if langs is not None:
        rec["langs"] = list(langs)
    if reply:
        parent = {"uri": f"at://{R.choice(DIDS)}/app.bsky.feed.post/{tid()}", "cid": "bafyreiparent"}
        rec["reply"] = {"root": parent, "parent": parent}
    return {"kind": "commit", "commit": {"rev": tid(), "operation": "create", "collection": "app.bsky.feed.post",
                                         "rkey": tid(), "record": rec, "cid": "bafyrei" + tid()}}


def noise():
    k = R.randrange(5)
    if k == 0:
        return {"kind": "identity", "identity": {"did": R.choice(DIDS), "handle": "someone.example.com", "seq": R.randrange(10**9)}}
    if k == 1:
        return {"kind": "account", "account": {"active": True, "did": R.choice(DIDS), "seq": R.randrange(10**9)}}
    if k == 2:
        return {"kind": "commit", "commit": {"rev": tid(), "operation": "delete", "collection": "app.bsky.feed.post", "rkey": tid()}}
    if k == 3:
        e = post("Microsoft edited this post to say Tim Cook was right about Apple.")
        e["commit"]["operation"] = "update"
        return e
    return {"kind": "commit", "commit": {"rev": tid(), "operation": "create", "collection": "app.bsky.feed.like", "rkey": tid(),
                                         "record": {"$type": "app.bsky.feed.like", "subject": {"uri": "at://x/app.bsky.feed.post/y"}}}}


def filtered(templates):
    k = R.randrange(6)
    fill = lambda t: t.format(city=R.choice(CITIES), person=R.choice(PEOPLE))
    if k == 0:
        return post(fill(R.choice(templates)), reply=True)
    if k == 1:
        langs, text = R.choice(FOREIGN)
        return post(text, langs)
    if k == 2:
        return post(fill(R.choice(templates)), langs=None)
    if k == 3:
        return post(R.choice(SHORT + [""]))
    if k == 4:
        return post(fill(GIVEAWAY))
    return noise()


def phase(templates, n_pass, n_other, first=None):
    passing = []
    if first:
        passing.append(post(first))
    while len(passing) < n_pass:
        t = R.choice(templates).format(city=R.choice(CITIES), person=R.choice(PEOPLE))
        passing.append(post(t, ["en-US"] if R.random() < 0.1 else ["en"]))
    others = [filtered(templates) for _ in range(n_other)]
    out = [passing[0]]
    rest = passing[1:] + others
    R.shuffle(rest)
    # keep the passing posts in their order (the sampler's positions are counted in that order)
    it = iter(passing[1:])
    return out + [next(it) if any(r is p for p in passing) else r for r in rest]


ev = phase(P1, 40, 26, first=XSS)
phase2_from = len(ev)
ev += phase(P2, 80, 54)
for i, e in enumerate(ev):
    e["did"] = R.choice(DIDS[:12]) if i % 3 else R.choice(DIDS)
    e["time_us"] = T0 + i * 7919
    events.append({k: e[k] for k in ("did", "time_us", "kind")} | {k: v for k, v in e.items() if k not in ("did", "time_us", "kind")})

doc = {
    "_doc": "Fictional Jetstream events for run/page_smoke2.mjs, written by run/make_jetstream_fixture.py. Phase 1 (events before "
            "phase2_from): every post that passes lang=en, no replies, minlen 20, exclude 'giveaway' mentions Microsoft; phase 2: none does.",
    "phase2_from": phase2_from,
    "events": events,
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n")
print(f"{len(events)} events, phase 2 from {phase2_from}, -> {OUT.name}")
