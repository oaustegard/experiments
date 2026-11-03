"""Yes/no bias check, both 8-bit models, greedy on onnxruntime CPU: 20 questions whose answer is yes, 20 no.
    python3 yesno.py   -> runs/yesno.jsonl and a summary
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from ort_gen import session, generate, pieces  # noqa: E402
from bridge import tok  # noqa: E402
from common import split_think  # noqa: E402
YES = ["Is a cat a mammal?", "Is the Sun a star?", "Is water wet?", "Is Paris in France?", "Is ice frozen water?",
       "Do birds have feathers?", "Is gold a metal?", "Is the Earth round?", "Do plants need sunlight?", "Is 10 greater than 3?",
       "Is a whale a mammal?", "Is Tokyo in Japan?", "Is the Pacific an ocean?", "Do humans need oxygen?", "Is a square a rectangle?",
       "Is 7 an odd number?", "Is steam hotter than ice?", "Is English spoken in Australia?", "Is a tomato a fruit?", "Do bees make honey?"]
NO = ["Is a dolphin a fish?", "Is the Moon a planet?", "Is fire cold?", "Is Berlin in Spain?", "Is a spider an insect?",
      "Can fish fly to the Moon?", "Is 3 greater than 10?", "Is the Sun a planet?", "Do snakes have legs?", "Is salt a metal?",
      "Is a bat a bird?", "Is Rome in Canada?", "Is 8 an odd number?", "Is glass a liquid at room temperature?", "Do cows lay eggs?",
      "Is the Atlantic a lake?", "Is a whale a fish?", "Is ice hotter than steam?", "Can humans breathe underwater without equipment?", "Is a circle a square?"]
t = tok(); P = pieces(t)
qs = [(q, "yes") for q in YES] + [(q, "no") for q in NO]
out = HERE / "runs/yesno.jsonl"
rows = []
for m in ("A8p_q8", "base_q8"):
    s = session(str(HERE.parent / "monad-weight-edit/export" / m), threads=4)
    for k in range(0, len(qs), 10):
        chunk = qs[k:k + 10]
        r = generate(s, [t.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False) for q, _ in chunk], 768, P)
        for (q, gold), tk in zip(chunk, r["tokens"] if isinstance(r, dict) else r[0]):
            ans = split_think(t.decode(tk))[1]
            first = re.match(r"\W*(yes|no)\b", ans, re.I)
            rows.append({"model": m, "q": q, "gold": gold, "said": first.group(1).lower() if first else None, "answer": ans[:300]})
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
for m in ("A8p_q8", "base_q8"):
    rs = [r for r in rows if r["model"] == m]
    print(m, "correct", sum(r["said"] == r["gold"] for r in rs), "/40;", "said yes", sum(r["said"] == "yes" for r in rs),
          "said no", sum(r["said"] == "no" for r in rs), "neither", sum(r["said"] is None for r in rs))
