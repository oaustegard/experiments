"""Phase 2: blind-guess critique and redraw loop.

  python3 phase2_critic.py blind  SRC ROUND   -> renders unlabelled PNGs, writes work/t2b_ROUND.jsonl
  python3 phase2_critic.py judge  SRC ROUND   -> writes work/t2j_ROUND.jsonl (needs blind results)
  python3 phase2_critic.py redraw SRC ROUND   -> writes work/t2r_ROUND.jsonl for verdict=redraw
  python3 phase2_critic.py merge  SRC ROUND DST -> DST gets SRC with redrawn figures swapped in where they validate and score better
SRC is a directory of constellation outputs (work/out1 for round 1).
"""
import json, sys, pathlib, shutil
from render import render, regions
from phase1_draw import validate, BIBLE, DRAW_RULES, star_table

W = pathlib.Path("work").resolve()

BLIND_SCHEMA = {"type": "object", "properties": {
    "guess": {"type": "string", "description": "what object, creature or scene the pale blue drawing shows, 3-12 words"},
    "alternatives": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["guess", "alternatives", "confidence"]}

JUDGE_SCHEMA = {"type": "object", "properties": {
    "match": {"type": "integer", "minimum": 1, "maximum": 5, "description": "how well the blind guess matches the intended subject (5 = same thing)"},
    "fit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "how well the drawing uses the stars: key stars on the right body parts, gold lines inside the figure"},
    "problems": {"type": "array", "items": {"type": "string"}},
    "advice": {"type": "string", "description": "concrete drawing instructions for a redraw, referring to coordinates and star ids"},
    "verdict": {"type": "string", "enum": ["keep", "redraw"]}}, "required": ["match", "fit", "problems", "advice", "verdict"]}

REDRAW_SCHEMA = {"type": "object", "properties": {k: v for k, v in __import__("phase1_draw").SCHEMA["properties"].items()
                                                  if k in ("anatomy", "lines", "figure")},
                 "required": ["anatomy", "lines", "figure"]}

def load(src, rid):
    p = pathlib.Path(src) / f"r{rid:03d}.json"
    return json.load(open(p))["output"] if p.exists() else None

def blind(src, rnd):
    png = W / f"blind{rnd}"; png.mkdir(exist_ok=True)
    with open(f"work/t2b_{rnd}.jsonl", "w") as f:
        for r in regions:
            o = load(src, r["rid"])
            if not o: continue
            ob = dict(o, anatomy=[])  # no labels
            render(r, ob, png / f"r{r['rid']:03d}.png", title=False)
            f.write(json.dumps(dict(id=f"r{r['rid']:03d}", system="You look at drawings and say plainly what they depict.",
                tools="Read", add_dir=[str(png)], schema=BLIND_SCHEMA,
                prompt=f"Read the image {png}/r{r['rid']:03d}.png. White dots are stars, gold lines join some of them, and the pale blue lines are an illustration drawn over the stars. "
                       "What does the pale blue illustration depict? Name the object, creature or scene as specifically as you can. If it is unrecognisable, say so and give confidence 1.")) + "\n")

def judge(src, rnd):
    png = W / f"png{rnd}"; png.mkdir(exist_ok=True)
    with open(f"work/t2j_{rnd}.jsonl", "w") as f:
        for r in regions:
            o = load(src, r["rid"]); b = W / f"out2b_{rnd}/r{r['rid']:03d}.json"
            if not o or not b.exists(): continue
            g = json.load(open(b))["output"]
            render(r, o, png / f"r{r['rid']:03d}.png")
            f.write(json.dumps(dict(id=f"r{r['rid']:03d}", system="You are a strict art director for a star atlas.",
                tools="Read", add_dir=[str(png)], schema=JUDGE_SCHEMA,
                prompt=f"""A constellation drawing is meant to show **{o['gloss']}**.
Key stars and the body part each should mark: {json.dumps(o['anatomy'])}

Someone who saw the drawing without its title guessed: "{g['guess']}" (alternatives: {g['alternatives']}; confidence {g['confidence']}/5).

Now read the labelled image {png}/r{r['rid']:03d}.png (labels show each key star's intended part; the frame is SVG-like, y grows downward).
Judge: does the blind guess match the intended subject? Do the stars sit on the right parts, is the figure the right way up, is it a real silhouette rather than a copy of the gold lines?
Verdict "redraw" if match <= 3 or fit <= 2; otherwise "keep". If redraw, give concrete advice (which shape to draw where, using coordinates and star ids).""")) + "\n")

def redraw(src, rnd):
    png = W / f"png{rnd}"
    n = 0
    with open(f"work/t2r_{rnd}.jsonl", "w") as f:
        for r in regions:
            o = load(src, r["rid"]); j = W / f"out2j_{rnd}/r{r['rid']:03d}.json"
            if not o or not j.exists(): continue
            jd = json.load(open(j))["output"]
            if jd["verdict"] != "redraw": continue
            g = json.load(open(W / f"out2b_{rnd}/r{r['rid']:03d}.json"))["output"]
            n += 1
            f.write(json.dumps(dict(id=f"r{r['rid']:03d}", system="You redraw one constellation figure for a star atlas so that it reads clearly.",
                tools="Read", add_dir=[str(png)], schema=REDRAW_SCHEMA,
                prompt=f"""{BIBLE}

---
# Redraw: {o['name']} — {o['gloss']}

The story (keep the subject, it is fixed): {o['myth']}

Stars available:
{star_table(r)}

The current drawing is at {png}/r{r['rid']:03d}.png — read it first.
Current anatomy: {json.dumps(o['anatomy'])}
Current lines: {json.dumps(o['lines'])}
A viewer who did not know the title guessed it was: "{g['guess']}".
The art director says: problems {json.dumps(jd['problems'])}; advice: {jd['advice']}

{DRAW_RULES}
Return a new anatomy, lines and figure for the same subject that a stranger would recognise.""")) + "\n")
    print(n, "redraws")

def merge(src, rnd, dst):
    dst = pathlib.Path(dst); dst.mkdir(exist_ok=True)
    swapped = 0
    for r in regions:
        p = pathlib.Path(src) / f"r{r['rid']:03d}.json"
        if not p.exists(): continue
        d = json.load(open(p))
        rd = W / f"out2r_{rnd}/r{r['rid']:03d}.json"
        if rd.exists():
            new = dict(d["output"], **json.load(open(rd))["output"])
            if not validate(r, new):
                d["output"] = new; d.setdefault("history", []).append(f"redrawn round {rnd}"); swapped += 1
        (dst / p.name).write_text(json.dumps(d))
    print("swapped", swapped)

if __name__ == "__main__":
    cmd, src, rnd = sys.argv[1], sys.argv[2], int(sys.argv[3])
    {"blind": blind, "judge": judge, "redraw": redraw}.get(cmd, lambda s, r: merge(s, r, sys.argv[4]))(src, rnd)
