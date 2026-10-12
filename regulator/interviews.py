"""Phase 1: interview transcripts.

  python3 interviews.py draft N        -> work/t_draft.jsonl  (N drafts per witness)
  python3 interviews.py check DIR      -> work/t_extract.jsonl + work/t_judge.jsonl for drafts in DIR
  python3 interviews.py score DIR      -> work/scores.json: which drafts pass both checks
  python3 interviews.py repair DIR     -> work/t_repair.jsonl: failing drafts with the checker's findings

A draft passes when (a) a blind reader, who never sees the fact cards, extracts
exactly the expected set of (timepiece, reading) pairs and no others, and (b) a
judge holding the fact cards marks every fact exact and finds no added claim
about times, places, movements, durations or sounds.
"""
import json, sys, pathlib, re
from facts import FACTS, TEXTURE, ROLE
from world import S, m, hhmm

W = pathlib.Path("work")
BIBLE = pathlib.Path("bible.md").read_text()
TP = ["salon clock", "kitchen clock", "hall clock (Morbier)", "atelier regulator",
      "Félix's watch", "Bastien's watch", "doctor's watch", "Lina's watch", "temple bell", "other"]
TPMAP = {"salon": "salon clock", "kitchen": "kitchen clock", "morbier": "hall clock (Morbier)",
         "regulator": "atelier regulator", "wF": "Félix's watch", "wB": "Bastien's watch",
         "wD": "doctor's watch", "wL": "Lina's watch"}

ANGLES = [
 "The inspector asks few questions; the witness talks at length and he steers gently.",
 "Brisk question and answer; the inspector takes the evening in order, one thing at a time.",
 "The witness begins with the shock of the night and the inspector brings them back to the evening.",
 "The inspector begins with the dead man and the witness's place in the household, then the evening.",
 "The witness is guarded at first; the inspector earns their trust and they open up.",
]

RELATIVE = {"L1", "D1"}  # cards state these as "N minutes behind/ahead", not as readings

def expected_readings(w):
    out = set()
    for x in S:
        if x["who"] == w and x["id"] not in RELATIVE:
            for a in x["atoms"]:
                if a[0] == "read":
                    out.add((TPMAP[a[2]], a[3]))
    return out

def draft_prompt(w, angle):
    facts = "\n".join(f"- {t}" for t in FACTS[w].values())
    return f"""Write the transcript of Inspector Léon Matthey's interview with {ROLE[w]}, taken down in the early hours of 1 January 1900.

THE WORLD
{BIBLE}

THE FACTS THIS WITNESS GIVES. Every one must appear in the transcript, in the witness's own words, with every clock or watch named and every reading exactly as given here. Write readings the way a person of 1899 says them ("twenty-eight minutes past ten by the salon clock", "a quarter to eleven by my watch"):
{facts}

WHAT ELSE THE WITNESS MAY TALK ABOUT
{TEXTURE[w]}

HARD RULES
1. No clock or watch reading, and no time of day, other than those in the facts. Do not round or paraphrase a reading.
2. No durations ("a few minutes", "a quarter of an hour", "soon after", "for a long while") unless a fact contains them.
3. No claim about where anyone was, who went where, or what anyone saw or heard, beyond the facts. Feelings, opinions, memories of the past and descriptions are fine.
4. Nobody does arithmetic about clocks, compares clocks or says a clock is fast or slow, except where a fact says so.
5. The inspector never tells the witness what others said, and never offers a theory.

SHAPE: {angle} 450 to 750 words. Format each line as "MATTHEY: ..." or "{LABEL[w]}: ..." Begin with one line in italics naming the witness and the hour of the interview by the inspector's account, e.g. "*Interview with ..., in the dining room. Second interview of the night.*" (give no clock time). Output only the transcript."""

EXTRACT_SCHEMA = {"type": "object", "properties": {
  "readings": {"type": "array", "items": {"type": "object", "properties": {
     "timepiece": {"type": "string", "enum": TP}, "shows": {"type": "string", "description": "HH:MM, 12-hour clock as read, e.g. 10:28"},
     "moment": {"type": "string"}}, "required": ["timepiece", "shows", "moment"]}},
  "durations": {"type": "array", "items": {"type": "string"}, "description": "every stated or implied length of time between two moments of that evening"},
  "whereabouts": {"type": "array", "items": {"type": "string"}, "description": "every claim about where a person was or went that evening, one per item"}},
  "required": ["readings", "durations", "whereabouts"]}

def extract_prompt(text):
    return f"""Below is a police interview transcript. List every clock or watch reading in it: each place where the witness says what a particular timepiece showed or struck at some moment. A temple bell striking an hour counts, with timepiece "temple bell". A clock striking an hour counts as that clock showing that hour exactly. Write each reading as H:MM on a 12-hour face (e.g. "twenty-three minutes to eleven" is 10:37). If you cannot tell which timepiece a reading came from, use "other". Also list every duration, and every claim about where anyone was or went. Be literal; include nothing that is not in the text.

TRANSCRIPT
{text}"""

JUDGE_SCHEMA = {"type": "object", "properties": {
  "facts": {"type": "array", "items": {"type": "object", "properties": {
     "id": {"type": "string"}, "verdict": {"type": "string", "enum": ["exact", "altered", "missing"]}, "note": {"type": "string"}},
     "required": ["id", "verdict", "note"]}},
  "added": {"type": "array", "items": {"type": "object", "properties": {
     "quote": {"type": "string"}, "kind": {"type": "string", "enum": ["time", "duration", "place", "movement", "sighting", "sound", "clock-comparison"]},
     "why": {"type": "string", "description": "what a detective could newly conclude from it"}},
     "required": ["quote", "kind", "why"]}},
  "quality": {"type": "integer", "minimum": 1, "maximum": 10}},
  "required": ["facts", "added", "quality"]}

def judge_prompt(w, text):
    facts = "\n".join(f"{k}: {t}" for k, t in FACTS[w].items())
    return f"""You are checking a fictional interview transcript against the fact sheet it was written from. The transcript is part of a fair-play logic puzzle, so a single wrong or extra detail can ruin it.

FACT SHEET
{facts}

For each fact: "exact" if the transcript states all of it with the same timepiece and the same reading or relation (wording may differ); "altered" if any timepiece, reading, order, person or place differs, or part is missing; "missing" if absent. Then list only the sentences that give a detective NEW information about the evening between dinner and the temple striking eleven, information the fact sheet does not already give: a new clock or watch reading or time of day; a length of time or an order between two events of the evening; a person being somewhere, or moving, at a moment the sheet does not cover; a new sound heard at a particular moment; a claim that some clock is fast or slow. Do NOT list anything that merely restates or colours a fact on the sheet, feelings, opinions, descriptions of people or rooms, habits ("he does it every year"), the past before that night, the quarrel at dinner, which music was played, or the finding of the body. When in doubt whether something is new, ask: could a detective work out a time, an order or a whereabouts from it that the sheet alone does not give? If not, leave it out. Then rate the writing 1-10 as period prose a reader would enjoy.

TRANSCRIPT
{text}"""

def norm(r):
    h, mm = map(int, r.split(":")); h = h + 12 if h < 12 else h
    return f"{h:02d}:{mm:02d}"

LABEL = dict(O="MME MONNIER", F="FÉLIX", L="MLLE RUEDI", D="DR VAUTHIER", B="GIRARD", R="ROSA")
SYS_W = "You write period fiction with exact fidelity to a fact sheet."
SYS_C = "You are a careful, literal reader."

def cmd_draft(n):
    with open(W / "t_draft.jsonl", "w") as f:
        for w in FACTS:
            for i in range(n):
                f.write(json.dumps(dict(id=f"{w}_{i}", system=SYS_W, prompt=draft_prompt(w, ANGLES[i % len(ANGLES)]))) + "\n")

def cmd_check(d):
    d = pathlib.Path(d)
    with open(W / f"t_extract_{d.name}.jsonl", "w") as fe, open(W / f"t_judge_{d.name}.jsonl", "w") as fj:
        for p in sorted(d.glob("*.json")):
            o = json.loads(p.read_text()); w = o["id"].split("_")[0]
            fe.write(json.dumps(dict(id=o["id"], system=SYS_C, prompt=extract_prompt(o["output"]), schema=EXTRACT_SCHEMA)) + "\n")
            fj.write(json.dumps(dict(id=o["id"], system=SYS_C, prompt=judge_prompt(w, o["output"]), schema=JUDGE_SCHEMA)) + "\n")

def evaluate(d):
    d = pathlib.Path(d); res = {}
    for p in sorted(d.glob("*.json")):
        o = json.loads(p.read_text()); did = o["id"]; w = did.split("_")[0]
        ex = W / f"x_{d.name}" / f"{did}.json"; ju = W / f"j_{d.name}" / f"{did}.json"
        if not ex.exists() or not ju.exists():
            continue
        X = json.loads(ex.read_text())["output"]; J = json.loads(ju.read_text())["output"]
        got = set()
        for r in X["readings"]:
            if r["timepiece"] == "temple bell" or (r["timepiece"] == "other" and r["shows"] in ("9:00", "10:00", "11:00")):
                continue
            try: got.add((r["timepiece"], norm(r["shows"])))
            except Exception: got.add((r["timepiece"], r["shows"]))
        exp = expected_readings(w)
        rel = {(TPMAP[a[2]], a[3]) for x in S if x["id"] in RELATIVE and x["who"] == w for a in x["atoms"] if a[0] == "read"}
        missing = sorted(exp - got); extra = sorted(got - exp - rel)
        bad_facts = [f for f in J["facts"] if f["verdict"] != "exact"]
        bad_add = [a for a in J["added"] if a["kind"] != "harmless"]
        words = len(o["output"].split())
        res[did] = dict(w=w, ok=not missing and not extra and not bad_facts and not bad_add,
                        missing=missing, extra=extra, bad_facts=bad_facts, bad_add=bad_add,
                        durations=X["durations"], quality=J["quality"], words=words)
    return res

def cmd_score(d):
    res = evaluate(d)
    json.dump(res, open(W / f"scores_{pathlib.Path(d).name}.json", "w"), indent=1, ensure_ascii=False)
    by = {}
    for k, v in res.items(): by.setdefault(v["w"], []).append(v["ok"])
    for w, oks in by.items(): print(w, f"{sum(oks)}/{len(oks)} pass")
    for k, v in res.items():
        if not v["ok"]:
            print(" ", k, "miss", v["missing"], "extra", v["extra"], "facts", [(f["id"], f["note"][:80]) for f in v["bad_facts"]],
                  "added", [(a["kind"], a["quote"][:70]) for a in v["bad_add"]])

def cmd_repair(d, out):
    d = pathlib.Path(d); res = evaluate(d)
    with open(W / "t_repair.jsonl", "w") as f:
        for did, v in res.items():
            if v["ok"]: continue
            text = json.loads((d / f"{did}.json").read_text())["output"]
            probs = []
            for t in v["missing"]: probs.append(f"A reader could not find this reading: {t[0]} showing {t[1]}. State it plainly, naming the timepiece.")
            for t in v["extra"]: probs.append(f"A reader found a reading that must not be there: {t[0]} showing {t[1]}. Remove it, or make the timepiece and reading unmistakable if it is one of the facts.")
            for x in v["bad_facts"]: probs.append(f"Fact {x['id']} is {x['verdict']}: {x['note']}")
            for a in v["bad_add"]: probs.append(f"Remove this added {a['kind']} claim: \"{a['quote']}\"")
            w = v["w"]
            prompt = draft_prompt(w, "Keep the existing shape and voice.") + "\n\nHERE IS A DRAFT THAT FAILED THE CHECK. Revise it, changing as little as possible, so that it fixes these problems and breaks no rule:\n" + "\n".join(f"- {p}" for p in probs) + f"\n\nDRAFT\n{text}\n\nOutput only the revised transcript."
            f.write(json.dumps(dict(id=did, system=SYS_W, prompt=prompt)) + "\n")

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "draft": cmd_draft(int(sys.argv[2]))
    elif c == "check": cmd_check(sys.argv[2])
    elif c == "score": cmd_score(sys.argv[2])
    elif c == "repair": cmd_repair(sys.argv[2], None)
