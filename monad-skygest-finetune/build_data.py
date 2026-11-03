"""Assemble training and evaluation files from data/gen/*.json, data/papers.json and the probe sets.

    python3 build_data.py
      data/train_papers.jsonl   10 SYNTH-style rows per paper
      data/heldout.jsonl        3 held-out questions per paper, with reference answer and key points
      data/eval_prompts.jsonl   every evaluation prompt: papers (held-out), facts (monad-chat's 30
                                one-line probes), dev (monad-weight-edit's 192-prompt dev fixture)

A paper counts as post-cutoff when it is dated 2025-11-01 or later: SYNTH and Monad were released in
November 2025, so nothing later can be in the weights.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MWE = experiment("monad-weight-edit")
sys.path.insert(0, str(MWE))
from common import gsm8k_gold, mmlu_question  # noqa: E402

CUTOFF = "2025-11-01"


def main():
    papers = {p["id"]: p for p in json.loads((HERE / "data/papers.json").read_text())}
    gens = sorted((HERE / "data/gen").glob("*.json"))
    tr, ho = [], []
    for g in gens:
        d = json.loads(g.read_text())
        p = papers[d["id"]]
        post = (p.get("date") or "0000") >= CUTOFF
        trq = {x["q"].strip().lower() for x in d["train"]}
        for i, x in enumerate(d["train"]):
            tr.append({"id": f"{g.stem}#t{i}", "paper": d["id"], "q": x["q"].strip(), "reasoning": x["reasoning"].strip(),
                       "answer": x["answer"].strip()})
        for i, x in enumerate(d["heldout"]):
            assert x["q"].strip().lower() not in trq
            ho.append({"id": f"{g.stem}#h{i}", "paper": d["id"], "title": p["title"], "date": p.get("date"),
                       "post_cutoff": post, "rank": p["rank"], "q": x["q"].strip(), "answer": x["answer"].strip(),
                       "key_points": x["key_points"]})
    # "seen": one training question per paper (its 3rd row), asked back verbatim: did the facts go in at all?
    seen = []
    for g in gens:
        d = json.loads(g.read_text())
        x = d["train"][2]
        p = papers[d["id"]]
        seen.append({"id": f"{g.stem}#s2", "paper": d["id"], "title": p["title"], "date": p.get("date"),
                     "post_cutoff": (p.get("date") or "0000") >= CUTOFF, "rank": p["rank"], "q": x["q"].strip(),
                     "answer": x["answer"].strip(), "key_points": []})
    for name, rows in (("train_papers", tr), ("heldout", ho), ("seen", seen)):
        with open(HERE / f"data/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ev = [{"set": "papers", "id": r["id"], "q": r["q"], "max_new": 768} for r in ho]
    ev += [{"set": "seen", "id": r["id"], "q": r["q"], "max_new": 768} for r in seen]
    for p in json.loads((experiment("monad-chat") / "probes_facts.json").read_text()):
        ev.append({"set": "facts", "id": p["id"], "q": p["q"], "check": p["check"], "max_new": 768})
    for x in json.loads((MWE / "data/fixture.json").read_text())["dev"]:
        r, t = x["row"], x["task"]
        q = mmlu_question(r) if t == "mmlu" else r["question"] if t == "gsm8k" else r["prompt"]
        e = {"set": "dev", "id": x["uid"], "task": t, "q": q, "max_new": 1536 if t == "ifeval" else 1024}
        if t == "mmlu":
            e["gold"] = "ABCD"[r["answer"]]
        if t == "gsm8k":
            e["gold"] = gsm8k_gold(r)
        ev.append(e)
    with open(HERE / "data/eval_prompts.jsonl", "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_post = sum(r["post_cutoff"] for r in ho) // 3
    print(f"{len(gens)} papers ({n_post} post-cutoff), {len(tr)} train rows, {len(ho)} held-out, {len(ev)} eval prompts")


if __name__ == "__main__":
    main()
