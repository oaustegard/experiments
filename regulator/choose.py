"""Phase 2: from drafts that passed both checks, hunt ambiguities and let a jury pick.

  python3 choose.py pool          -> work/pool/*.json, work/t_hunt.jsonl, work/t_jury.jsonl
  python3 choose.py pick          -> work/chosen.json (best jury rank among drafts with no fatal ambiguity)
"""
import json, sys, pathlib, random
from facts import FACTS, ROLE
W = pathlib.Path("work")

HUNT_SCHEMA = {"type": "object", "properties": {"issues": {"type": "array", "items": {"type": "object", "properties": {
    "quote": {"type": "string"}, "fact": {"type": "string"}, "misreading": {"type": "string"},
    "severity": {"type": "string", "enum": ["fatal", "minor"]}}, "required": ["quote", "fact", "misreading", "severity"]}}},
    "required": ["issues"]}
JURY_SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "string"}},
    "best_reason": {"type": "string"}}, "required": ["ranking", "best_reason"]}

def pool():
    P = W / "pool"; P.mkdir(exist_ok=True)
    for rnd in ["d0", "d1"]:
        sc = json.load(open(W / f"scores_{rnd}.json"))
        for did, v in sc.items():
            if v["ok"]:
                o = json.loads((W / rnd / f"{did}.json").read_text())
                (P / f"{rnd}_{did}.json").write_text(json.dumps(dict(id=f"{rnd}_{did}", w=v["w"], text=o["output"], quality=v["quality"]), ensure_ascii=False))
    items = [json.loads(p.read_text()) for p in sorted(P.glob("*.json"))]
    with open(W / "t_hunt.jsonl", "w") as f:
        for it in items:
            facts = "\n".join(f"{k}: {t}" for k, t in FACTS[it["w"]].items())
            for r in range(2):
                f.write(json.dumps(dict(id=f"{it['id']}__h{r}", system="You are an adversarial proofreader for logic puzzles.", schema=HUNT_SCHEMA, prompt=f"""This interview transcript is one witness statement in a fair-play murder puzzle whose solution turns on exact clock readings. The writer was told to convey the facts below and nothing more. Your job is to find places where an honest, careful solver could take the transcript to mean something DIFFERENT from a fact: a reading attached to the wrong clock or watch, a reading that could be heard two ways ("half ten", "ten to"), events whose order is unclear, a "he" or "she" that could be the wrong person, a hint of a duration or a whereabouts the sheet does not give. "fatal" means a solver could reasonably reach a different reading of a fact; "minor" means awkward but not really misleading. Report nothing if the transcript is clean; do not invent problems.

FACT SHEET
{facts}

TRANSCRIPT
{it['text']}""")) + "\n")
    with open(W / "t_jury.jsonl", "w") as f:
        for w in FACTS:
            mine = [it for it in items if it["w"] == w]
            for j in range(5):
                random.seed(j * 7 + ord(w)); order = mine[:]; random.shuffle(order)
                labels = {chr(65 + i): it["id"] for i, it in enumerate(order)}
                body = "\n\n".join(f"=== VERSION {k} ===\n{json.loads((P / (v + '.json')).read_text())['text']}" for k, v in labels.items())
                json.dump(labels, open(W / f"jury_labels_{w}_{j}.json", "w"))
                f.write(json.dumps(dict(id=f"{w}__j{j}", system="You are a demanding fiction editor.", schema=JURY_SCHEMA, prompt=f"""Here are {len(labels)} versions of the same police interview with {ROLE[w]}, from a murder mystery set in La Chaux-de-Fonds on the night of 31 December 1899. They carry the same facts. Rank them from best to worst as writing a reader would enjoy: a distinct, believable voice; period feel without pastiche; tension; and the facts arriving naturally rather than as a recited list. Return the ranking as a list of version letters.

{body}""")) + "\n")

def pick():
    P = W / "pool"; items = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in P.glob("*.json")}
    fatal = {k: [] for k in items}
    for p in (W / "hunt").glob("*.json"):
        o = json.loads(p.read_text()); did = o["id"].split("__")[0]
        fatal[did] += [i for i in o["output"]["issues"] if i["severity"] == "fatal"]
    score = {k: 0.0 for k in items}
    for p in (W / "jury").glob("*.json"):
        o = json.loads(p.read_text()); w, j = o["id"].split("__"); j = j[1:]
        labels = json.load(open(W / f"jury_labels_{w}_{j}.json"))
        rk = [r.strip().upper()[:1] for r in o["output"]["ranking"]]
        n = len(labels)
        for pos, lab in enumerate(rk):
            if lab in labels: score[labels[lab]] += (n - pos) / n
    chosen = {}
    for w in FACTS:
        cands = sorted([k for k in items if items[k]["w"] == w], key=lambda k: -score[k])
        print(w, [(k, round(score[k], 2), len(fatal[k])) for k in cands])
        clean = [k for k in cands if not fatal[k]]
        chosen[w] = dict(id=(clean or cands)[0], fatal=fatal[(clean or cands)[0]])
    json.dump(dict(chosen=chosen, fatal=fatal, score=score), open(W / "chosen.json", "w"), indent=1, ensure_ascii=False)
    for w, c in chosen.items():
        print(w, c["id"], "fatal issues:", [(i["quote"][:60], i["misreading"][:100]) for i in c["fatal"]])

if __name__ == "__main__":
    {"pool": pool, "pick": pick}[sys.argv[1]]()
