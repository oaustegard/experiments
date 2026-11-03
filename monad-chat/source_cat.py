"""Answering from a handed source: a cat paragraph appended to the question in SYNTH's <source_1> format,
both 8-bit models, greedy, onnxruntime CPU.  python3 source_cat.py -> runs/source_cat.jsonl"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from ort_gen import session, generate, pieces  # noqa: E402
from bridge import tok  # noqa: E402
from common import split_think  # noqa: E402
SRC = ("The domestic cat (Felis catus) is a small carnivorous mammal. It is the only domesticated species of the family Felidae. "
       "Cats were domesticated in the Near East around 7500 BC. Adult cats typically weigh 4 to 5 kilograms. Cats are obligate "
       "carnivores and are valued by humans for companionship and for hunting rodents.")
QS = ["Is a cat a mammal?", "Where were cats domesticated?", "How much does an adult cat weigh?", "What do cats eat?"]
t = tok(); P = pieces(t); rows = []
for m in ("A8p_q8", "base_q8"):
    s = session(str(HERE.parent / "monad-weight-edit/export" / m), threads=4)
    for src in (True, False):
        ps = [f"<|im_start|>user\n{q}" + (f"\n\n<source_1>\n{SRC}\n</source_1>" if src else "") + "<|im_end|>\n<|im_start|>assistant\n<think>\n" for q in QS]
        r = generate(s, [t.encode(p, add_special_tokens=False) for p in ps], 768, P)
        for q, tk in zip(QS, r["tokens"] if isinstance(r, dict) else r[0]):
            rows.append({"model": m, "source": src, "q": q, "answer": split_think(t.decode(tk))[1]})
(HERE / "runs/source_cat.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
for x in rows: print(x["model"], x["source"], x["q"], "|", x["answer"][:100].replace("\n", " "))
