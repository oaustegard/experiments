import sys, json, random, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/austegard/Projects/ane-search")
from common import *; import kb
import os; B = os.path.dirname(os.path.abspath(__file__))
R = kb.Reader(); random.seed(11)
ok = [i for i in range(R.n) if R.live[i] and 350 < len(R.meta[i]["text"]) and prose(R.meta[i]["text"] * 3) and R.meta[i]["text"].count("|") < 12]
pick = sorted(random.sample(ok, 240))
for k in range(3):
    with open(f"{B}/passages_{k}.md", "w") as f:
        for i in pick[k::3]: f.write(f"\n\n=== PASSAGE {i} ===\nfile: {R.meta[i]['path']}\nheading: {R.meta[i]['head']}\n\n{R.meta[i]['text'][:1100]}\n")
json.dump(pick, open(f"{B}/pick.json", "w")); print(len(ok), len(pick))
if "--embed" in sys.argv:
    emb = Embedder(); full = lambda m: m["path"] + (" — " + m["head"] if m["head"] else "") + "\n" + m["text"]
    V = np.zeros((R.n, 768), np.float16)
    for i in range(R.n):
        if R.live[i]: V[i] = emb.doc_ids(emb.tok.encode(full(R.meta[i])).ids)
    np.save(f"{B}/float768.npy", V); print("embedded")
