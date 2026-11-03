"""run7 on the 96x96 E1M9 grids, one 25-grid batch at a time (resumable: each batch is saved as it finishes).
Merges into results.json as e1m9_96/run7 when all four batches are done."""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from handtrm import model_from, load_bin
from eval_doom import evaluate
torch.set_num_threads(4)
X = np.load(HERE / "data/e1m9_96/x.npy")[:100]; Y = np.load(HERE / "data/e1m9_96/y.npy")[:100]
m = model_from(load_bin(str(HERE / "doom/bot/trm")))
part = HERE / "run7_96_parts.json"; done = json.load(open(part)) if part.exists() else {}
for b in range(0, 100, 25):
    if str(b) in done: continue
    ex, _ = evaluate(m, X[b:b + 25], Y[b:b + 25], 96, 16, 25, False)
    done[str(b)] = ex.tolist(); json.dump(done, open(part, "w")); print("batch", b, flush=True)
ex = np.concatenate([np.array(done[str(b)]) for b in range(0, 100, 25)], 1)
res = json.load(open(HERE / "results.json"))
res["e1m9_96/run7"] = {"n": 100, "reachable": float((Y.sum(1) > 0).mean()),
    "exact_by_step": {str(s + 1): round(float(ex[s].mean()), 4) for s in range(16)}, "bot_sup": 8, "exact_at_bot_sup": round(float(ex[7].mean()), 4)}
json.dump(res, open(HERE / "results.json", "w"), indent=1); print(res["e1m9_96/run7"])
