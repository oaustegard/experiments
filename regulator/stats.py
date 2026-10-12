"""Count every child call and its cost, by stage, into work/stats.json (with an HTML table for the page)."""
import json, glob, collections, pathlib
STAGE = {"d0": "draft", "d1": "repair", "x_d0": "blind extract", "x_d1": "blind extract", "j_d0": "fidelity judge", "j_d1": "fidelity judge",
         "hunt": "ambiguity hunt", "jury": "jury", "fin": "final triple check", "det": "detectives", "conf": "confession", "confj": "confession jury"}
rows = collections.OrderedDict(); tot = collections.Counter()
for root, label in [("work_v2/v1", "v1"), ("work_v2", "v2"), ("work", "v3")]:
    for d in sorted(pathlib.Path(root).iterdir()):
        if not d.is_dir() or d.name not in STAGE: continue
        for p in d.glob("*.json"):
            o = json.loads(p.read_text())
            if "cost" not in o: continue
            model = "Opus" if p.stem.startswith("opus") else "Sonnet" if p.stem.startswith("sonnet") else "Haiku"
            k = (label, STAGE[d.name], model); rows.setdefault(k, [0, 0.0, 0.0]); r = rows[k]
            r[0] += 1; r[1] += o["cost"]; r[2] += o.get("secs", 0)
            tot[model + "_n"] += 1; tot[model + "_cost"] += o["cost"]
agg = collections.OrderedDict()
for (v, st, mdl), r in rows.items():
    key = (st, mdl); a = agg.setdefault(key, [0, 0.0, set()]); a[0] += r[0]; a[1] += r[1]; a[2].add(v)
html = ['<div class="statwrap"><table class="stats"><tr><th>Stage</th><th>Model</th><th>Calls</th><th>Cost</th><th>Rounds</th></tr>']
for (st, mdl), (n, c, vs) in agg.items():
    html.append(f'<tr><td>{st}</td><td>{mdl}</td><td class="n">{n}</td><td class="n">${c:.2f}</td><td>{", ".join(sorted(vs))}</td></tr>')
html.append(f'<tr><th>All</th><th></th><th class="n">{sum(a[0] for a in agg.values())}</th><th class="n">${sum(a[1] for a in agg.values()):.2f}</th><th></th></tr></table></div>')
out = dict(rows={" / ".join(k): v for k, v in rows.items()}, totals=dict(tot), html="\n".join(html))
json.dump(out, open("work/stats.json", "w"), indent=1)
print(dict(tot)); print(sum(a[0] for a in agg.values()), "calls")

# detectives per version
def score(d):
    res = collections.defaultdict(lambda: [0, 0])
    for p in glob.glob(f"{d}/*.json"):
        o = json.loads(open(p).read())["output"]; mdl = pathlib.Path(p).stem.split("_")[0]
        ok = o["murderer"].startswith("Dr") and o["time"][-5:] in ("22:38", "10:38")
        res[mdl][0] += ok; res[mdl][1] += 1
    return dict(res)
ver = {"v1": score("work_v2/v1/det"), "v2": score("work_v2/det"), "v3": score("work/det")}
out["versions"] = ver
f = lambda v, mdl: (lambda r: f"{r[0]} of {r[1]}" if r else "none")(ver[v].get(mdl))
out["intro"] = f"""<p>The case was built backwards from a ground truth: a minute-by-minute evening, eight timepieces with fixed errors, and {len(__import__('world').S)} statements written as constraints, each one a reading of a named clock at a named moment, an order of events, or a person in a room. The z3 solver checks that the honest accounts are true of that evening and the doctor's is not, that exactly one person can be the murderer, and that the blow falls on exactly one minute. It also finds, for each innocent, the small sets of statements that cannot all be true if that person did it. Those are the chains listed above.</p>
<p>Claude Haiku 5.5 wrote the interviews from first-person fact cards, five drafts per witness. Each draft was checked before anyone read it. A blind reader who never saw the facts listed every clock reading it could find, and code compared that list with the expected one. A judge holding the facts marked each fact exact, altered or missing, and listed anything a detective could use that the facts did not give. Failed drafts were repaired once. Adversarial readers then hunted the survivors for ways a careful solver could misread them, and a jury of five Haiku ranked them as writing. I edited the six winners by hand, and the checks ran three more times on the edited text. They caught one bug I had introduced, a "he" that read as the dead man.</p>
<p>The first two versions were too easy. On the second, Haiku solved {f('v2','haiku')} and Sonnet {f('v2','sonnet')}: every witness had read a clock against the temple bell, so the doctor's lie fell out of simple arithmetic. The third version removed most of those direct readings, so errors have to be carried from clock to clock through shared sounds (the piano, the hall clock's strikes, the repeater), and it added Rosa's mistaken belief about her own clock. On this version, Haiku solved {f('v3','haiku')}, Sonnet {f('v3','sonnet')} and Opus {f('v3','opus')}. One more Haiku detective was still working after fifteen minutes and was stopped. The rest took four to seven minutes each, against about a minute on the second version.</p>"""
json.dump(out, open("work/stats.json", "w"), indent=1)
print(ver)
