"""Phase 3: verify the edited finals (3 blind extractions, 3 fidelity judges, 3
ambiguity hunters each) and send the whole dossier to detectives.

  python3 finalcheck.py tasks      -> work/t_final.jsonl, work/t_det.jsonl
  python3 finalcheck.py report     -> work/final_report.json
"""
import json, sys, pathlib, collections
import interviews as I
from facts import FACTS, ROLE
W = pathlib.Path("work"); F = pathlib.Path("final")

def dossier():
    parts = [pathlib.Path("premise.md").read_text()]
    for p in sorted(F.glob("*.md")):
        parts.append(p.read_text())
    return "\n\n---\n\n".join(parts)

DET_SCHEMA = {"type": "object", "properties": {
    "murderer": {"type": "string", "enum": list(ROLE.values())},
    "time": {"type": "string", "description": "true time of the blow, HH:MM 24-hour"},
    "reasoning": {"type": "string"}, "confidence": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["murderer", "time", "reasoning", "confidence"]}

def tasks():
    finals = {p.stem.split("_")[1]: p.read_text() for p in F.glob("*.md")}
    hunt = json.loads(open(W / "t_hunt.jsonl").readline())
    with open(W / "t_final.jsonl", "w") as f:
        for w, text in finals.items():
            for r in range(3):
                f.write(json.dumps(dict(id=f"x__{w}__{r}", system=I.SYS_C, prompt=I.extract_prompt(text), schema=I.EXTRACT_SCHEMA)) + "\n")
                f.write(json.dumps(dict(id=f"j__{w}__{r}", system=I.SYS_C, prompt=I.judge_prompt(w, text), schema=I.JUDGE_SCHEMA)) + "\n")
                facts = "\n".join(f"{k}: {t}" for k, t in FACTS[w].items())
                hp = hunt["prompt"]; head = hp[:hp.index("FACT SHEET")]
                f.write(json.dumps(dict(id=f"h__{w}__{r}", system=hunt["system"], schema=hunt["schema"],
                                        prompt=head + f"FACT SHEET\n{facts}\n\nTRANSCRIPT\n{text}")) + "\n")
    D = dossier()
    with open(W / "t_det.jsonl", "w") as f:
        for i in range(20):
            f.write(json.dumps(dict(id=f"haiku_{i}", system="You are a detective solving a fair-play puzzle.", schema=DET_SCHEMA, timeout=900,
                                    prompt=D + "\n\n---\n\nSolve the case. Work carefully from the testimony and the inspector's rules, then answer.")) + "\n")
        for i in range(5):
            f.write(json.dumps(dict(id=f"sonnet_{i}", model="claude-sonnet-5-5", system="You are a detective solving a fair-play puzzle.", schema=DET_SCHEMA, timeout=900,
                                    prompt=D + "\n\n---\n\nSolve the case. Work carefully from the testimony and the inspector's rules, then answer.")) + "\n")
    print(len(D.split()), "words in dossier")

def report():
    out = collections.defaultdict(dict)
    for p in (W / "fin").glob("*.json"):
        o = json.loads(p.read_text()); kind, w, r = o["id"].split("__"); out[w][kind + r] = o["output"]
    rep = {}
    for w in sorted(out):
        exp = I.expected_readings(w); res = []
        for r in range(3):
            X = out[w].get(f"x{r}"); J = out[w].get(f"j{r}"); H = out[w].get(f"h{r}")
            got = set()
            for rd in X["readings"]:
                if rd["timepiece"] == "temple bell" or (rd["timepiece"] == "other" and rd["shows"] in ("9:00", "10:00", "11:00")): continue
                got.add((rd["timepiece"], I.norm(rd["shows"])))
            rel = {(I.TPMAP[a[2]], a[3]) for x in I.S if x["id"] in I.RELATIVE and x["who"] == w for a in x["atoms"] if a[0] == "read"}
            res.append(dict(missing=sorted(exp - got), extra=sorted(got - exp - rel),
                            facts=[(f["id"], f["verdict"], f["note"]) for f in J["facts"] if f["verdict"] != "exact"],
                            added=[(a["kind"], a["quote"], a["why"]) for a in J["added"]],
                            fatal=[(h["quote"], h["misreading"]) for h in H["issues"] if h["severity"] == "fatal"]))
        rep[w] = res
        print(f"== {w}")
        for i, x in enumerate(res):
            print(f"  run{i}: miss={x['missing']} extra={x['extra']} facts={x['facts']}")
            for a in x["added"]: print("     added:", a[0], "|", a[1][:90], "|", a[2][:90])
            for h in x["fatal"]: print("     FATAL:", h[0][:90], "|", h[1][:140])
    json.dump(rep, open(W / "final_report.json", "w"), indent=1, ensure_ascii=False)

if __name__ == "__main__":
    {"tasks": tasks, "report": report}[sys.argv[1]]()
