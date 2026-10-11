"""Phase 1: one Haiku agent per region invents and draws a constellation.

  python3 phase1_draw.py tasks            -> work/t1.jsonl
  python3 phase1_draw.py check            -> validates work/out1, writes repair tasks to work/t1r.jsonl
"""
import json, glob, random, re, sys, pathlib
from collections import defaultdict, deque

BIBLE = open("bible.md").read()
regions = json.load(open("work/regions.json"))
HOUSE_KEY = {"The Spindle": "spindle", "The Windward Reach": "windward", "The Lamp Road": "lamp",
             "The Leeward Reach": "leeward", "The Unflown": "unflown"}
PANTHEON = ["Aun", "Hesk", "Tavo", "Ennu", "Sil", "Orrun", "Kael", "The Gulls"]

SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string", "description": "Ilvari name, 2-3 syllables, obeying the bible's sound rules"},
    "gloss": {"type": "string", "description": "English name, e.g. 'The Split Oar'"},
    "anatomy": {"type": "array", "description": "the key stars and which part of the figure each one marks",
                "items": {"type": "object", "properties": {"star": {"type": "integer"}, "part": {"type": "string"}}, "required": ["star", "part"]}},
    "lines": {"type": "array", "description": "constellation stick-figure edges, each a pair of star ids",
              "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}},
    "figure": {"type": "array", "description": "outline drawing: a list of polylines, each a list of [x,y] points in the same 0-1000 frame",
               "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}},
    "star_names": {"type": "array", "description": "Ilvari names for the 1-2 brightest stars you used",
                   "items": {"type": "object", "properties": {"star": {"type": "integer"}, "name": {"type": "string"}, "gloss": {"type": "string"}},
                             "required": ["star", "name", "gloss"]}},
    "myth": {"type": "string", "description": "the story, 110-170 words"},
    "verse": {"type": "string", "description": "a three-line verse of 5, 7 and 5 syllables, lines separated by newlines"},
    "gods": {"type": "array", "items": {"type": "string", "enum": PANTHEON}},
}, "required": ["name", "gloss", "anatomy", "lines", "figure", "star_names", "myth", "verse", "gods"]}

SYSTEM = "You invent and draw one constellation of a fictional culture's sky. Follow the world bible exactly. Return only the structured output."

def star_table(r):
    rows = ["id | x | y | magnitude (lower = brighter)"]
    rows += [f"{s['id']} | {s['x']} | {s['y']} | {s['mag']:.2f}" for s in r["stars"]]
    return "\n".join(rows)

DRAW_RULES = """How to draw:
1. Look at the star coordinates. The frame is like an SVG canvas, 1000 x 1000: x grows to the right, y grows DOWNWARD, so y=0 is the top of the picture. Sketch the pattern in your head first: where are the bright stars, what lines and clusters do they make? Which way up does the figure stand?
2. Choose the concept whose shape the stars actually suggest. You may adapt it.
3. Pick 5-14 stars as the figure's key points (head, eye, wingtip, blade, handle, prow...). Use at least two of the four brightest. Record each in `anatomy`.
4. `lines`: connect those stars into a single connected stick figure (every used star reachable from every other). Edges should not cross each other.
5. `figure`: the illustration, like an engraving in an old star atlas. It is NOT a copy of the lines: it is the thing's silhouette with body and volume (a bird's rounded body, wing feathers, a hull's curve and a sail's belly, a person's head, arms and clothing), drawn so the key stars fall at the right places on it (the eye star sits inside the drawn eye, the wingtip star at the drawn wingtip). 3-10 polylines, 8-40 points each, 60-200 points in total, same coordinate frame. Smooth curves need many points. Close a shape by repeating its first point. Keep it inside 0-1000. A reader should be able to name the object from the drawing alone.
"""

def build_tasks():
    pools = defaultdict(list)
    for f in glob.glob("work/out0/*.json"):
        d = json.load(open(f)); h = d["id"].split("-")[0]
        seen = {c["gloss"].lower() for c in pools[h]}
        for c in d["output"]["concepts"]:
            if c["gloss"].lower() not in seen:
                pools[h].append(c); seen.add(c["gloss"].lower())
    random.seed(11)
    for p in pools.values(): random.shuffle(p)
    cursor = defaultdict(int)
    with open("work/t1.jsonl", "w") as f:
        for r in regions:
            h = HOUSE_KEY[r["house"]]; pool = pools[h]
            seeds = [pool[(cursor[h] + i) % len(pool)] for i in range(3)]; cursor[h] += 3
            r["seeds"] = seeds
            prompt = f"""{BIBLE}

---
# Your patch of sky

House: **{r['house']}**. This patch is about {r['width_deg']}° across.
Stars you may use (sorted brightest first):

{star_table(r)}

Concepts other people suggested for this house (pick the one that fits these stars, or adapt one):
""" + "\n".join(f"- {s['gloss']}: {s['hook']}" for s in seeds) + f"""

{DRAW_RULES}
Then write the myth (110-170 words, the bible's register, name at least one god from the pantheon), and a three-line 5-7-5 verse. Name the one or two brightest stars you used in Ilvari."""
            f.write(json.dumps(dict(id=f"r{r['rid']:03d}", system=SYSTEM, schema=SCHEMA, prompt=prompt)) + "\n")
    json.dump(regions, open("work/regions.json", "w"))
    print("wrote", len(regions), "tasks")

NAME_RE = re.compile(r"^[ktslrnmvhaeiouáéíóú' -]+$", re.I)

def validate(r, o):
    errs = []
    ids = {s["id"] for s in r["stars"]}
    bright4 = [s["id"] for s in r["stars"][:4]]
    if not NAME_RE.match(o["name"]):
        errs.append(f"name '{o['name']}' uses letters outside the Ilvari alphabet (allowed: k t s l r n m v h, th, sk, a e i o u, á é í ó ú)")
    bad = [e for e in o["lines"] if e[0] not in ids or e[1] not in ids or e[0] == e[1]]
    if bad: errs.append(f"lines use star ids not in your table or self-loops: {bad[:5]}")
    used = {i for e in o["lines"] for i in e if i in ids}
    if len(used) < 4: errs.append(f"only {len(used)} stars used in lines; use 5-14")
    if len(used) > 16: errs.append(f"{len(used)} stars used; use at most 14")
    if not set(bright4) & used: errs.append(f"use at least two of the four brightest stars {bright4}")
    adj = defaultdict(set)
    for a, b in o["lines"]:
        adj[a].add(b); adj[b].add(a)
    if used:
        start = next(iter(used)); seen = {start}; q = deque([start])
        while q:
            for nb in adj[q.popleft()]:
                if nb not in seen: seen.add(nb); q.append(nb)
        if seen != used: errs.append(f"lines are not one connected figure: stars {sorted(used - seen)[:6]} are cut off from the rest")
    pts = [p for pl in o["figure"] for p in pl]
    if len(o["figure"]) < 1 or len(pts) < 8: errs.append("figure drawing is missing or too small; give 2-8 polylines")
    if any(not (-150 <= x <= 1150 and -150 <= y <= 1150) for x, y in pts): errs.append("figure points fall far outside the 0-1000 frame")
    if any(a["star"] not in ids for a in o["anatomy"]): errs.append("anatomy names star ids not in your table")
    if any(s["star"] not in ids for s in o["star_names"]): errs.append("star_names uses ids not in your table")
    for s in o["star_names"]:
        if not NAME_RE.match(s["name"]): errs.append(f"star name '{s['name']}' breaks the Ilvari alphabet")
    w = len(o["myth"].split())
    if not 80 <= w <= 230: errs.append(f"myth is {w} words; write 110-170")
    if len([l for l in o["verse"].split("\n") if l.strip()]) != 3: errs.append("verse must be exactly three lines")
    return errs

def check(outdir="work/out1"):
    bad = {}
    for r in regions:
        p = pathlib.Path(f"{outdir}/r{r['rid']:03d}.json")
        if not p.exists(): bad[r["rid"]] = ["missing"]; continue
        e = validate(r, json.load(open(p))["output"])
        if e: bad[r["rid"]] = e
    print(f"{len(regions)-len(bad)}/{len(regions)} valid")
    for k, v in list(bad.items())[:12]: print(k, v)
    return bad

if __name__ == "__main__":
    if sys.argv[1] == "tasks": build_tasks()
    elif sys.argv[1] == "check":
        bad = check()
        tasks = {json.loads(l)["id"]: json.loads(l) for l in open("work/t1.jsonl")}
        with open("work/t1r.jsonl", "w") as f:
            for rid, errs in bad.items():
                t = tasks[f"r{rid:03d}"]; prev = pathlib.Path(f"work/out1/r{rid:03d}.json")
                if prev.exists():
                    t["prompt"] += ("\n\n---\nA previous attempt was rejected by the checker:\n- " + "\n- ".join(errs) +
                                    "\n\nPrevious attempt:\n" + json.dumps(json.load(open(prev))["output"]) + "\n\nFix every problem and return the full corrected constellation.")
                f.write(json.dumps(t) + "\n")
        print(len(bad), "repair tasks")
