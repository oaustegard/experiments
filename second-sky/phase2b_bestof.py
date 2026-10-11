"""Best-of selection across critique rounds, then best-of-N redraws for figures still unrecognised.

  python3 phase2b_bestof.py select           -> work/best/ (best judged version per figure)
  python3 phase2b_bestof.py redraw N          -> work/t5r.jsonl (N independent redraws per weak figure)
  python3 phase2b_bestof.py blind             -> renders candidates, work/t5b.jsonl
  python3 phase2b_bestof.py judge             -> work/t5j.jsonl
  python3 phase2b_bestof.py final             -> work/final/ (winner per figure, with its blind guess)
"""
import json, sys, pathlib
from render import render, regions
from phase1_draw import validate, BIBLE, DRAW_RULES, star_table
from phase2_critic import BLIND_SCHEMA, JUDGE_SCHEMA, REDRAW_SCHEMA

W = pathlib.Path("work").resolve()
VERSIONS = [("out1", "out2j_1", "out2b_1", "png1"), ("art1", "out2j_2", "out2b_2", "png2"), ("art2", "out2j_3", "out2b_3", "png3")]
RB = {r["rid"]: r for r in regions}
J = lambda p: json.load(open(p))

def select():
    dst = W / "best"; dst.mkdir(exist_ok=True)
    weak = 0
    for r in regions:
        n = f"r{r['rid']:03d}.json"; cands = []
        for i, (s, j, b, png) in enumerate(VERSIONS):
            if (W / j / n).exists():
                jd = J(W / j / n)["output"]
                cands.append(((jd["match"], jd["fit"], i), dict(J(W / s / n), judge=jd, guess=J(W / b / n)["output"]["guess"], png=str(W / png / n.replace(".json", ".png")))))
        key, best = max(cands, key=lambda c: c[0])
        best["version"] = key[2]
        weak += key[0] <= 3
        (dst / n).write_text(json.dumps(best))
    print("weak (best match <= 3):", weak)

def redraw(N):
    with open(W / "t5r.jsonl", "w") as f:
        for p in sorted((W / "best").glob("r*.json")):
            d = J(p); o = d["output"]; jd = d["judge"]
            if jd["match"] >= 4: continue
            r = RB[int(p.stem[1:])]
            for k in range(N):
                f.write(json.dumps(dict(id=f"{p.stem}_{k}", system="You redraw one constellation figure for a star atlas so that a stranger recognises it at once.",
                    tools="Read", add_dir=[str(pathlib.Path(d['png']).parent)], schema=REDRAW_SCHEMA,
                    prompt=f"""{BIBLE}

---
# Redraw: {o['name']} — {o['gloss']}

The story (the subject is fixed): {o['myth']}

Stars available:
{star_table(r)}

The current drawing is at {d['png']} — read it first.
Current anatomy: {json.dumps(o['anatomy'])}
A viewer who did not know the title guessed it was: "{d['guess']}".
The art director says: problems {json.dumps(jd['problems'])}; advice: {jd['advice']}

{DRAW_RULES}
Draw it fresh. Think about the single most recognisable silhouette of this subject (its characteristic outline, the way a child would draw it) and fit that silhouette to the stars. Simplify: fewer, bolder shapes read better than many small ones.""")) + "\n")
    print(sum(1 for _ in open(W / "t5r.jsonl")), "redraw tasks")

def cand_iter():
    for p in sorted((W / "out5r").glob("r*_*.json")):
        rid = int(p.stem[1:4]); base = J(W / "best" / f"r{rid:03d}.json")
        new = dict(base["output"], **J(p)["output"])
        if not validate(RB[rid], new):
            yield p.stem, rid, new

def blind():
    png = W / "blind5"; png.mkdir(exist_ok=True); cd = W / "cand5"; cd.mkdir(exist_ok=True)
    with open(W / "t5b.jsonl", "w") as f:
        for cid, rid, o in cand_iter():
            (cd / f"{cid}.json").write_text(json.dumps(o))
            render(RB[rid], dict(o, anatomy=[]), png / f"{cid}.png", title=False)
            f.write(json.dumps(dict(id=cid, system="You look at drawings and say plainly what they depict.", tools="Read", add_dir=[str(png)], schema=BLIND_SCHEMA,
                prompt=f"Read the image {png}/{cid}.png. White dots are stars, gold lines join some of them, and the pale blue lines are an illustration drawn over the stars. "
                       "What does the pale blue illustration depict? Name the object, creature or scene as specifically as you can. If it is unrecognisable, say so and give confidence 1.")) + "\n")
    print(sum(1 for _ in open(W / "t5b.jsonl")), "valid candidates")

def judge():
    png = W / "png5"; png.mkdir(exist_ok=True)
    with open(W / "t5j.jsonl", "w") as f:
        for p in sorted((W / "cand5").glob("*.json")):
            cid = p.stem; rid = int(cid[1:4]); o = J(p)
            b = W / "out5b" / f"{cid}.json"
            if not b.exists(): continue
            g = J(b)["output"]
            render(RB[rid], o, png / f"{cid}.png")
            f.write(json.dumps(dict(id=cid, system="You are a strict art director for a star atlas.", tools="Read", add_dir=[str(png)], schema=JUDGE_SCHEMA,
                prompt=f"""A constellation drawing is meant to show **{o['gloss']}**.
Key stars and the body part each should mark: {json.dumps(o['anatomy'])}

Someone who saw the drawing without its title guessed: "{g['guess']}" (alternatives: {g['alternatives']}; confidence {g['confidence']}/5).

Now read the labelled image {png}/{cid}.png (labels show each key star's intended part; the frame is SVG-like, y grows downward).
Judge: does the blind guess match the intended subject? Do the stars sit on the right parts, is the figure the right way up, is it a real silhouette rather than a copy of the gold lines?
Verdict "redraw" if match <= 3 or fit <= 2; otherwise "keep".""")) + "\n")

def final():
    dst = W / "final"; dst.mkdir(exist_ok=True); improved = 0; rec = 0
    for p in sorted((W / "best").glob("r*.json")):
        d = J(p); best = ((d["judge"]["match"], d["judge"]["fit"]), d["output"], d["guess"], d["judge"]["match"], d.get("history", []))
        for c in sorted((W / "out5j").glob(f"{p.stem}_*.json")):
            jd = J(c)["output"]
            if (jd["match"], jd["fit"]) > best[0]:
                best = ((jd["match"], jd["fit"]), J(W / "cand5" / c.name), J(W / "out5b" / c.name)["output"]["guess"], jd["match"], d.get("history", []) + ["best-of-n redraw"])
        if best[4] != d.get("history", []): improved += 1
        rec += best[3] >= 4
        (dst / p.name).write_text(json.dumps(dict(id=p.stem, output=best[1], guess=best[2], match=best[3], history=best[4])))
    print("improved by best-of-n:", improved, "| recognised (match>=4):", rec, "of", len(regions))

if __name__ == "__main__":
    a = sys.argv[1]
    {"select": select, "blind": blind, "judge": judge, "final": final}.get(a, lambda: redraw(int(sys.argv[2])))()
