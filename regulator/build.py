"""Collect everything the page needs into work/case.json."""
import json, re, pathlib, glob, collections
import world as Wd
from facts import ROLE

def md(text):
    """Tiny markdown: headings, bullets, numbered lists, bold, italics, paragraphs."""
    out, para, lst = [], [], None
    def inline(s):
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    def flush():
        nonlocal para, lst
        if para: out.append("<p>" + inline(" ".join(para)) + "</p>"); para = []
        if lst: out.append(f"<{lst[0]}>" + "".join(f"<li>{inline(i)}</li>" for i in lst[1]) + f"</{lst[0]}>"); lst = None
    for line in text.split("\n"):
        if line.startswith("#"):
            flush(); n = len(line) - len(line.lstrip("#")); out.append(f"<h{n+1}>{inline(line.lstrip('# '))}</h{n+1}>")
        elif re.match(r"^(- |\d+\. )", line):
            if para: flush()
            kind = "ol" if line[0].isdigit() else "ul"
            if not lst or lst[0] != kind: flush(); lst = (kind, [])
            lst[1].append(re.sub(r"^(- |\d+\. )", "", line))
        elif line.startswith("   ") and lst:
            lst[1][-1] += " " + line.strip()
        elif not line.strip():
            flush()
        else:
            if lst: flush()
            para.append(line.strip())
    flush(); return "\n".join(out)

NUM = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:-(?:one|two|three|four|five|six|seven|eight|nine))?|twenty|twenty-eight|twenty-nine|thirty(?:-two)?|a quarter|half)"
HOUR = r"(?:nine|ten|eleven)"
PAT = re.compile(rf"\b((?:{NUM} (?:minutes? )?(?:past|to) {HOUR})|(?:struck|strike|strikes) {HOUR}|{HOUR} o'clock)\b", re.I)

def transcript(text):
    lines = [l for l in text.strip().split("\n") if l.strip()]
    head = lines[0].strip("*"); body = []
    for l in lines[1:]:
        m = re.match(r"^([A-ZÉ ]+):\s*(.*)$", l)
        who, said = (m.group(1), m.group(2)) if m else ("", l)
        said = re.sub(r"\*(.+?)\*", r"<em>\1</em>", said)
        said = PAT.sub(r'<mark class="t">\1</mark>', said)
        body.append(dict(who=who, text=said))
    return dict(head=head, lines=body)

def main():
    v = json.load(open("work/verify.json"))
    finals = {p.stem.split("_")[1]: transcript(p.read_text()) for p in sorted(pathlib.Path("final").glob("*.md"))}
    order = [p.stem.split("_")[1] for p in sorted(pathlib.Path("final").glob("*.md"))]
    det = collections.defaultdict(lambda: collections.Counter())
    for p in glob.glob("work/det/*.json"):
        o = json.load(open(p))["output"]; model = pathlib.Path(p).stem.split("_")[0]
        who = o["murderer"].split(",")[0]; t = o["time"][-5:]
        if t[:2] in ("09", "10", "11") and int(t[:2]) < 12: t = f"{int(t[:2]) + 12}:{t[3:]}"
        det[model][f"{who} at {t}"] += 1
    case = dict(
        premise=md(pathlib.Path("premise.md").read_text().split("\n", 3)[3]),
        people=Wd.PEOPLE, roles=ROLE, victim=Wd.VICTIM, order=order,
        transcripts=finals,
        statements=[dict(id=s["id"], who=s["who"], gloss=s["gloss"], atoms=s["atoms"], lie=s["lie"]) for s in Wd.S if s["who"]],
        beliefs=Wd.BELIEFS,
        events={k: lab for k, (lab, _) in Wd.EVENTS.items()},
        clocks={k: v_ for k, v_ in Wd.CLOCKS.items() if k != "salon"},
        truth=dict(off={k: v_ for k, v_ in Wd.TRUE_OFF.items() if k != "salon"},
                   events={k: t for k, (_, t) in Wd.EVENTS.items()}),
        solution=Wd.SOLUTION, trains={k: v_[:3] for k, v_ in v["trains"].items()},
        ntrains={k: len(v_) for k, v_ in v["trains"].items()},
        detectives={k: dict(c) for k, c in det.items()},
        confession=pathlib.Path("confession.md").read_text() if pathlib.Path("confession.md").exists() else "",
        stats=json.load(open("work/stats.json")) if pathlib.Path("work/stats.json").exists() else {},
    )
    json.dump(case, open("work/case.json", "w"), ensure_ascii=False)
    print("case.json", len(json.dumps(case, ensure_ascii=False)) // 1024, "KB;", sum(len(t["lines"]) for t in finals.values()), "lines;",
          sum(l["text"].count('class="t"') for t in finals.values() for l in t["lines"]), "time marks")

if __name__ == "__main__":
    main()
