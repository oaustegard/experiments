"""Build labelled eval sets for a typed-decision harness from dettin-68m's gold labels.

    python3 run/build_sets.py     # -> data/papers.jsonl.gz, data/sections.jsonl.gz

Rows are in the format `deciding-with-confidence/scripts/decide.py eval` reads:
{id, source, input, questions, labels}. A question carries a label only where Sonnet 5.5 and Opus 5.5,
labelling independently, agreed (the article-part question of a section is labelled from the article's own
heading). The text is CC BY, from Europe PMC; ids are PMC ids and section ids from dettin-68m's corpus.
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
D68 = HERE.parent / "dettin-68m"
sys.path.insert(0, str(D68 / "run"))
import paper  # noqa: E402
from suite import SECTION_QS  # noqa: E402


def question(name, q):
    if q["type"] == "noul":
        return {"type": "predicate", "name": name, "instructions": q["instructions"]}
    if isinstance(q["criteria"], list):
        return {"type": "score", "name": name, "instructions": q["instructions"],
                "levels": [{"label": f"level {i}", "description": d} for i, d in enumerate(q["criteria"])]}
    return {"type": "choice", "name": name, "instructions": q["instructions"],
            "choices": [{"value": k, "description": d} for k, d in q["criteria"].items()]}


def label(q, v):
    if q["type"] == "noul":
        return v == "true"
    return int(v) if isinstance(q["criteria"], list) else v


def read(p):
    with open(p) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def lab(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def write(name, rows):
    with gzip.open(HERE / "data" / f"{name}.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = sum(len(r["labels"]) for r in rows)
    print(f"{name}: {len(rows)} items, {n} labelled questions")


def main():
    arts = paper.articles()
    picks = json.loads((D68 / "data" / "paper_picks.json").read_text())
    gs, go = read(D68 / "data" / "paper_gold_sonnet.jsonl"), read(D68 / "data" / "paper_gold_opus.jsonl")
    rows = []
    for a in picks["test"]:
        if a not in gs or a not in go:
            continue
        labels = {k: label(q, lab(go[a][k])) for k, q in paper.QS.items() if lab(gs[a][k]) == lab(go[a][k])}
        rows.append({"id": a, "source": "Europe PMC, CC BY; gold = Sonnet 5.5 and Opus 5.5 agree on the full article",
                     "input": paper.text_of(arts[a]), "questions": [question(k, q) for k, q in paper.QS.items()],
                     "labels": labels})
    write("papers", rows)

    gold = json.loads((D68 / "data" / "gold.json").read_text())["sections"]
    text = {u["sid"]: u["text"] for units in arts.values() for u in units}
    rows = []
    for sid, g in gold.items():
        labels = {}
        for k, q in SECTION_QS.items():
            if k == "role":
                if g["heading_role"]:
                    labels[k] = g["heading_role"]
            elif g["sonnet"][k] == g["opus"][k]:
                labels[k] = label(q, g["opus"][k])
        rows.append({"id": sid, "source": "Europe PMC, CC BY; gold = Sonnet 5.5 and Opus 5.5 agree on the section; "
                                          "role from the article's heading",
                     "input": text[sid], "questions": [question(k, q) for k, q in SECTION_QS.items()], "labels": labels})
    write("sections", rows)


if __name__ == "__main__":
    main()
