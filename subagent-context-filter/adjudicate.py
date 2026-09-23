"""Blind adjudication of regex misses, with the adjudicator validated first.

The fact regexes were written before any deliverable existed and some are too
literal ("3868" misses "3.9 s"; "3 failed, 5 passed" misses "5 passed, 3
failed"). Every (task, arm, fact) the regex missed goes to an adjudicator that
sees the fact, the transcript evidence and the deliverable, never the arm.

Validation items are mixed into the same batches, indistinguishable:
  positives: regex hits (the fact is there)                 -> TPR
  negatives: the fact checked against a deliverable for a
             DIFFERENT task from the same session (same topic,
             the fact should be absent)                      -> TNR
Output: data/adjudication.json
"""
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import claude  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ARMS = ["none", "jev1", "jev4", "full", "brief"]
BATCH = 20

PROMPT = """You are checking deliverables for specific facts. For each item you get a FACT name, the EVIDENCE (the exact text from the source where the fact appears) and a DELIVERABLE. Decide whether the deliverable states that same fact correctly. Equivalent formats count (3868 ms = 3.9 s; "5 passed, 3 failed" = "3 failed, 5 passed"; a short hash that matches the start of the full one). A different value, a hedge without the value, or saying the fact is missing do not count.

Reply with ONLY a JSON array, one object per item, in this order of keys: {{"id": ..., "critique": "one sentence", "present": true|false}}

ITEMS:
{items}"""


def main():
    rnd = random.Random(7)
    tasks = {t["task_id"]: t for f in glob.glob(os.path.join(DATA, "tasks", "*.json")) for t in json.load(open(f))}
    res = {}
    for f in glob.glob(os.path.join(DATA, "results", "*.json")):
        tid, arm = os.path.basename(f)[:-5].rsplit(".", 1)
        res[(tid, arm)] = json.load(open(f))
    items, kinds = [], {}
    for (tid, arm), r in res.items():
        for i, hit in enumerate(r.get("facts") or []):
            if not hit and arm != "none":
                items.append({"key": [tid, arm, i]}); kinds[len(items) - 1] = "miss"
    hits = [(tid, arm, i) for (tid, arm), r in res.items() if arm != "none"
            for i, h in enumerate(r.get("facts") or []) if h]
    for tid, arm, i in rnd.sample(hits, 40):
        items.append({"key": [tid, arm, i]}); kinds[len(items) - 1] = "pos"
    by_sess = {}
    for tid in tasks:
        by_sess.setdefault(tid.rsplit("-", 1)[0], []).append(tid)
    for _ in range(40):
        tid = rnd.choice(list(tasks)); other = rnd.choice([x for x in by_sess[tid.rsplit("-", 1)[0]] if x != tid])
        i = rnd.randrange(len(tasks[tid]["facts"]))
        items.append({"key": [tid, "full", i], "deliverable_of": other}); kinds[len(items) - 1] = "neg"
    order = list(range(len(items))); rnd.shuffle(order)
    verdicts = {}
    for b in range(0, len(order), BATCH):
        batch = order[b:b + BATCH]
        lines = []
        for j in batch:
            tid, arm, i = items[j]["key"]
            fact = tasks[tid]["facts"][i]
            src = items[j].get("deliverable_of", tid)
            deliv = res[(src, arm)]["deliverable"][:2500]
            lines.append(json.dumps({"id": j, "fact": fact["name"], "evidence": fact["evidence"], "deliverable": deliv}))
        out = claude(PROMPT.format(items="\n".join(lines)))
        m = re.search(r"\[.*\]", out["result"], re.S)
        try:
            for v in json.loads(m.group(0)):
                verdicts[int(v["id"])] = bool(v["present"])
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            print("batch parse failed", b, type(e).__name__, file=sys.stderr)
        print(f"batch {b // BATCH + 1}/{(len(order) + BATCH - 1) // BATCH}: {len(verdicts)} verdicts", flush=True)
    pos = [verdicts[j] for j, k in kinds.items() if k == "pos" and j in verdicts]
    neg = [not verdicts[j] for j, k in kinds.items() if k == "neg" and j in verdicts]
    upgrades = [items[j]["key"] for j, k in kinds.items() if k == "miss" and verdicts.get(j)]
    out = {"tpr": sum(pos) / len(pos), "tnr": sum(neg) / len(neg), "n_pos": len(pos), "n_neg": len(neg),
           "misses": sum(1 for k in kinds.values() if k == "miss"),
           "misses_judged": sum(1 for j, k in kinds.items() if k == "miss" and j in verdicts),
           "upgrades": upgrades}
    json.dump(out, open(os.path.join(DATA, "adjudication.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "upgrades"}), len(upgrades), "upgrades")


if __name__ == "__main__":
    main()
