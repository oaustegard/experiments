"""Can Monad be made to say it does not know? Loop-fixed 8-bit, greedy, onnxruntime CPU.
1. probes_facts.json plain: first confidence mark in the trace (● ◐ ○) vs correctness.
2. Same questions with " If you are not sure, say that you don't know." appended.
3. Cat source, questions the source does not answer.
    python3 idk.py -> runs/idk.jsonl
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from ort_gen import session, generate, pieces  # noqa: E402
from bridge import tok  # noqa: E402
from common import split_think  # noqa: E402
SRC = json.loads(open(HERE / "runs/source_cat.jsonl").readline())["answer"] and ("The domestic cat (Felis catus) is a small carnivorous mammal. It is the only domesticated species of the family Felidae. Cats were domesticated in the Near East around 7500 BC. Adult cats typically weigh 4 to 5 kilograms. Cats are obligate carnivores and are valued by humans for companionship and for hunting rodents.")
IDK = re.compile(r"(don't know|do not know|not sure|unsure|uncertain|cannot (provide|confirm|determine|answer)|can't (provide|confirm)|don't have|not (provided|mentioned|included|specified) in|does not (mention|specify|provide|include)|doesn't (mention|specify|provide|include))", re.I)
t = tok(); P = pieces(t)
s = session(str(HERE.parent / "monad-weight-edit/export/A8p_q8"), threads=4)
facts = json.loads((HERE / "probes_facts.json").read_text())
items = [("plain", p["id"], p["q"], p["check"]) for p in facts]
items += [("idk_suffix", p["id"], p["q"] + " If you are not sure, say that you don't know.", p["check"]) for p in facts]
for q in ["How long do cats live?", "What colour are most cats?", "How many whiskers does a cat have?", "Who was the first person to keep a cat as a pet?"]:
    items.append(("source_missing", q, f"{q}\n\n<source_1>\n{SRC}\n</source_1>", None))
rows = []
for k in range(0, len(items), 10):
    chunk = items[k:k + 10]
    r = generate(s, [t.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False) for _, _, q, _ in chunk], 900, P)
    for (arm, pid, q, chk), tk in zip(chunk, r["tokens"] if isinstance(r, dict) else r[0]):
        think, ans, _ = split_think(t.decode(tk))
        m = re.search(r"[●◐○]", think)
        rows.append({"arm": arm, "probe": pid, "q": q, "ok": bool(chk and re.search(chk, ans, re.I)), "mark": m.group() if m else None,
                     "idk": bool(IDK.search(ans)), "answer": ans[:400]})
    print(k, flush=True)
(HERE / "runs/idk.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
from collections import Counter
for arm in ("plain", "idk_suffix"):
    rs = [x for x in rows if x["arm"] == arm]
    print(arm, "right", sum(x["ok"] for x in rs), "idk", sum(x["idk"] for x in rs), "idk&wrong", sum(x["idk"] and not x["ok"] for x in rs), "idk&right", sum(x["idk"] and x["ok"] for x in rs))
    for mk in ("●", "◐", "○", None):
        sub = [x for x in rs if x["mark"] == mk]
        if sub: print("  first mark", mk, f"{sum(x['ok'] for x in sub)}/{len(sub)} right")
for x in rows:
    if x["arm"] == "source_missing": print("SRC-MISSING", x["probe"], "| idk:", x["idk"], "|", x["answer"][:160].replace("\n", " "))
