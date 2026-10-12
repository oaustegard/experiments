"""The doctor's letter, for the reveal. 6 drafts, 7 jurors.
  python3 confession.py draft | jury | pick"""
import json, sys, pathlib, random
W = pathlib.Path("work"); BIBLE = pathlib.Path("bible.md").read_text()
TRUTH = """What really happened (all true times, by the temple bell):
- In 1889 Vauthier treated Jules Monnier, Aurèle's younger brother and Félix's father, with laudanum, raised the dose too far in a bad night, and Jules died of it. Vauthier signed the certificate as heart failure. Nobody knew.
- In December 1899, sorting his brother's papers for the new century, Aurèle found Jules's last letter, which described the doses. On the evening of 31 December, in the hall before dinner, Aurèle told Vauthier he would tell Félix at midnight, and the authorities after. "At the turn of the century I settle my accounts."
- After dinner Vauthier played piquet with Odile in the salon. At 22:33 he rose to fetch his cigars and asked her to play for him; the piano would cover his steps. He went up the two flights to the atelier (passing the shut library door at 22:35).
- Lina had been sent down at 22:20. Aurèle was alone at his bench under the long windows, with the repeater. Vauthier struck him once from behind with the brass mallet at 22:38. Falling, Aurèle pressed the repeater's slide: ten low notes, two doubled, eight high: 10:38. Vauthier heard it and counted it, as anyone in that town would.
- He stopped the regulator's pendulum and moved its hands back to five past ten, an hour when he had sat at cards with Odile in plain view. He came down at 22:41, took the cigar case from his coat in the hall and was back in the salon at 22:46. Odile stopped playing when he came in. The hall clock struck its eleven a moment later.
- To the inspector he said he had gone out at a quarter to eleven by his watch, after the repeater would have sounded. He did not know that Félix, below, and Girard, in the street, had both heard it, or that the kitchen had heard the piano begin.
- His watch was two minutes fast all evening. He told the truth about everything else."""

SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"}}, "required": ["ranking", "why"]}

def draft():
    angles = ["He writes as a physician writing a case history of himself.",
              "He writes about time: the habit of clocks in that town, and the minute he could not move.",
              "He writes to Félix as much as to the inspector, though the letter is addressed to Matthey.",
              "He writes plainly, almost coldly, in order, and only lets feeling through at the end.",
              "He writes about the piano: what Odile played, and what it covered.",
              "He writes about the repeater: what he heard, and that he counted it."]
    with open(W / "t_conf.jsonl", "w") as f:
        for i, a in enumerate(angles):
            f.write(json.dumps(dict(id=f"c{i}", system="You write restrained period fiction.", prompt=f"""{BIBLE}

{TRUTH}

Write the letter Dr Casimir Vauthier leaves for Inspector Matthey at dawn on 1 January 1900, before the inspector comes back to arrest him. {a}

Rules: every fact you use must agree with the account above; add no new times, readings or movements. No melodrama, no villain's speech, no self-pity at length. He is a cultivated man of 58 who has been found out and knows exactly how. He may use one Latin tag at most. 350 to 550 words. Sign it. Output only the letter.""")) + "\n")

def jury():
    D = {p.stem: json.loads(p.read_text())["output"] for p in sorted((W / "conf").glob("*.json"))}
    keys = sorted(D)
    with open(W / "t_confj.jsonl", "w") as f:
        for j in range(7):
            random.seed(j); order = keys[:]; random.shuffle(order)
            lab = {chr(65 + i): k for i, k in enumerate(order)}; json.dump(lab, open(W / f"confj_{j}.json", "w"))
            body = "\n\n".join(f"=== LETTER {l} ===\n{D[k]}" for l, k in lab.items())
            f.write(json.dumps(dict(id=f"j{j}", system="You are a demanding literary editor.", schema=SCHEMA, prompt=f"These are {len(lab)} drafts of a murderer's confession letter that closes a period detective story (La Chaux-de-Fonds, 1 January 1900). Rank them best to worst: restraint, a voice you believe, the clockwork of the crime made clear without lecturing, and an ending that lands. Return letters.\n\n{body}")) + "\n")

def pick():
    sc = {}
    for p in (W / "confj").glob("*.json"):
        o = json.loads(p.read_text()); lab = json.load(open(W / f"conf{o['id'].replace('j', 'j_')}.json"))
        r = [x.strip()[:1].upper() for x in o["output"]["ranking"]]
        for i, l in enumerate(r):
            if l in lab: sc[lab[l]] = sc.get(lab[l], 0) + len(lab) - i
    print(sorted(sc.items(), key=lambda x: -x[1]))

if __name__ == "__main__":
    {"draft": draft, "jury": jury, "pick": pick}[sys.argv[1]]()
