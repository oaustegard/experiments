"""Phase 3: reweave each myth so it knows its neighbours, and say how to find it in the sky.
Phase 4: competing epics for the tour, judged by a jury.

  python3 phase3_weave.py weave SRC        -> work/t3.jsonl
  python3 phase3_weave.py epic SRC         -> work/t4.jsonl (6 composers)
  python3 phase3_weave.py jury SRC         -> work/t4j.jsonl (7 jurors over valid epics)
  python3 phase3_weave.py pick SRC         -> work/epic.json
"""
import json, math, sys, pathlib, glob, statistics
from collections import Counter
from phase1_draw import BIBLE, NAME_RE

regions = json.load(open("work/regions.json"))
RB = {r["rid"]: r for r in regions}

def load_all(src):
    out = {}
    for r in regions:
        p = pathlib.Path(src) / f"r{r['rid']:03d}.json"
        if p.exists(): out[r["rid"]] = json.load(open(p))["output"]
    return out

def centre(r, o):
    pos = {s["id"]: s for s in r["stars"]}
    used = {i for e in o["lines"] for i in e if i in pos}
    vs = [(math.cos(math.radians(pos[i]["dec"])) * math.cos(math.radians(pos[i]["ra"])),
           math.cos(math.radians(pos[i]["dec"])) * math.sin(math.radians(pos[i]["ra"])), math.sin(math.radians(pos[i]["dec"]))) for i in used]
    x, y, z = (sum(v[k] for v in vs) / len(vs) for k in range(3))
    n = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.atan2(y, x)) % 360, math.degrees(math.asin(z / n))

def sep_pa(a, b):
    ra1, d1, ra2, d2 = map(math.radians, (*a, *b))
    sep = math.degrees(math.acos(max(-1, min(1, math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra2 - ra1)))))
    pa = math.degrees(math.atan2(math.sin(ra2 - ra1), math.cos(d1) * math.tan(d2) - math.sin(d1) * math.cos(ra2 - ra1))) % 360
    return sep, ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((pa + 22.5) // 45) % 8]

def summary(o):
    s = o["myth"].split(". ")
    return ". ".join(s[:2]).strip()[:260]

WEAVE_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "myth": {"type": "string", "description": "130-190 words"},
    "finding": {"type": "string", "description": "one sentence telling a person on the islands how to find it, from a neighbour"}},
    "required": ["name", "myth", "finding"]}

def weave(src):
    al = load_all(src)
    names = Counter(o["name"].lower() for o in al.values()); glosses = Counter(o["gloss"].lower() for o in al.values())
    seen_n, seen_g = set(), set()
    with open("work/t3.jsonl", "w") as f:
        for rid, o in al.items():
            r = RB[rid]; c = centre(r, o)
            nb = []
            for n in r["neighbours"]:
                if n not in al: continue
                sep, d = sep_pa(c, centre(RB[n], al[n]))
                bright = min(RB[n]["stars"], key=lambda s: s["mag"])["mag"]
                nb.append(f"- **{al[n]['name']}** — {al[n]['gloss']} ({RB[n]['house']}), {sep:.0f}° to the {d}; brightest star magnitude {bright:.1f}. Its story: {summary(al[n])}")
            clash = ""
            if names[o["name"].lower()] > 1 and o["name"].lower() in seen_n:
                clash += f"\nYour Ilvari name '{o['name']}' is already used by another constellation: coin a new one that follows the sound rules."
            if glosses[o["gloss"].lower()] > 1 and o["gloss"].lower() in seen_g:
                clash += f"\nAnother constellation is also called '{o['gloss']}'. Keep your subject but make the story's angle distinct from a generic version."
            seen_n.add(o["name"].lower()); seen_g.add(o["gloss"].lower())
            f.write(json.dumps(dict(id=f"r{rid:03d}", system="You are the keeper of a fictional culture's sky-lore. Follow the world bible exactly.", schema=WEAVE_SCHEMA,
                prompt=f"""{BIBLE}

---
# {o['name']} — {o['gloss']} ({r['house']})

Its myth as first told:
{o['myth']}

Its neighbours in the sky (directions are as seen on the sky, distances in degrees; a fist at arm's length is about 10°):
""" + "\n".join(nb) + f"""

Retell the myth in 130-190 words so that it knows where it lives: weave in at least two neighbours by their English names, in ways consistent with their stories and with the real directions (if the Gull lies to the north, the story can have it watch from above). Keep the original subject, gods and best images; the neighbours should change the story's meaning a little, not just be listed. Keep the bible's register.
Then write `finding`: one plain sentence telling an islander how to find it, starting from a neighbour, using the real direction and distance in fists or hand-spans (a hand-span is about 20°). For a constellation of the Unflown, say how sailors far south find it.
Return `name` unchanged unless told otherwise.{clash}""")) + "\n")

EPIC_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"}, "prologue": {"type": "string", "description": "60-100 words"},
    "stops": {"type": "array", "minItems": 12, "maxItems": 16, "items": {"type": "object", "properties": {
        "id": {"type": "integer"}, "text": {"type": "string", "description": "50-90 words"}}, "required": ["id", "text"]}}},
    "required": ["title", "prologue", "stops"]}

def catalogue(src):
    al = load_all(src); rows = []
    for rid, o in sorted(al.items()):
        ra, dec = centre(RB[rid], o)
        rows.append(f"{rid} | {o['name']} — {o['gloss']} | {RB[rid]['house']} | RA {ra:.0f}° Dec {dec:+.0f}° | {summary(o)[:160]}")
    return "\n".join(rows), al

ANGLES = ["Tell it from Sil the Moon's side, searching for her brother.",
          "Tell it as Hesk the Reel counts what the cutting cost.",
          "Tell it as a children's walk across the sky on a winter night.",
          "Tell it as Aun's own confession, late in his life.",
          "Tell it from the Gulls, who carried the news between sea and sky.",
          "Tell it as the year turns: start at the Spindle, go out along the Lamp Road, end somewhere strange."]

def epic(src):
    cat, _ = catalogue(src)
    with open("work/t4.jsonl", "w") as f:
        for i, angle in enumerate(ANGLES):
            f.write(json.dumps(dict(id=f"epic{i}", system="You compose the central epic of a fictional culture's sky. Follow the world bible exactly.", schema=EPIC_SCHEMA,
                prompt=f"""{BIBLE}

---
Here is every constellation in the Ilvari sky (id | name | house | position | story opening):
{cat}

Compose **the Cutting of the String**, the Ilvari epic, as a guided tour of 12-16 of these constellations. {angle}
Rules: each stop is one constellation id from the list, no repeats; consecutive stops must be within 45° of each other on the sky so the tour flies short hops; each stop's text (50-90 words) tells the next part of the epic through that figure, consistent with its own story. A title, and a prologue of 60-100 words. Spare, maritime register.""")) + "\n")

def ang(al, a, b):
    return sep_pa(centre(RB[a], al[a]), centre(RB[b], al[b]))[0]

def valid_epics(src):
    al = load_all(src); good = []
    for p in sorted(glob.glob("work/out4/epic*.json")):
        e = json.load(open(p))["output"]; ids = [s["id"] for s in e["stops"]]
        errs = []
        if any(i not in al for i in ids): errs.append("bad id")
        if len(set(ids)) != len(ids): errs.append("repeat")
        if not errs:
            hops = [ang(al, a, b) for a, b in zip(ids, ids[1:])]
            if max(hops) > 60: errs.append(f"long hop {max(hops):.0f}")
        print(pathlib.Path(p).stem, e["title"], len(ids), errs or "ok")
        if not errs: good.append((pathlib.Path(p).stem, e))
    return good, al

JURY_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {"type": "object", "properties": {
    "key": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 10}, "why": {"type": "string"}},
    "required": ["key", "score", "why"]}}}, "required": ["scores"]}

def jury(src):
    good, al = valid_epics(src)
    def show(k, e):
        return f"## Epic {k}: {e['title']}\n{e['prologue']}\n" + "\n".join(f"{n+1}. [{al[s['id']]['gloss']}] {s['text']}" for n, s in enumerate(e["stops"]))
    body = "\n\n".join(show(k, e) for k, e in good)
    with open("work/t4j.jsonl", "w") as f:
        for j in range(7):
            f.write(json.dumps(dict(id=f"juror{j}", system="You are a demanding literary judge.", schema=JURY_SCHEMA,
                prompt=f"""{BIBLE}\n\n---\nThese are competing versions of the Ilvari epic, each told as a tour across the constellations. Score each 1-10 for: does it work as a story with an arc and an ending, is it in the bible's register (spare, concrete, tender, a little uncanny; no fantasy cliché), and does each stop earn its place. Use the full range. Keys are the epic ids.\n\n{body}""")) + "\n")

def pick(src):
    good, al = valid_epics(src)
    sc = {k: [] for k, _ in good}
    for p in glob.glob("work/out4j/*.json"):
        for s in json.load(open(p))["output"]["scores"]:
            k = s["key"] if s["key"].startswith("epic") else "epic" + s["key"].split()[-1]
            if k in sc: sc[k].append(s["score"])
    ranked = sorted(sc, key=lambda k: -statistics.mean(sc[k] or [0]))
    for k in ranked: print(k, sc[k], round(statistics.mean(sc[k] or [0]), 2))
    best = dict(good)[ranked[0]]
    best["jury"] = {k: round(statistics.mean(v), 2) for k, v in sc.items() if v}
    json.dump(best, open("work/epic.json", "w"))
    print("picked", ranked[0], best["title"])

if __name__ == "__main__":
    {"weave": weave, "epic": epic, "jury": jury, "pick": pick}[sys.argv[1]](sys.argv[2])
