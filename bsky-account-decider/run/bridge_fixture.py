"""Write out/bridge/fixtures.json for web/bridge_job.js: 8 test accounts (other people's public posts, kept
out of git), the question files, and run/pack_page.py's NumPy reference probabilities for account_type."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "browser-decider" / "run")); sys.path.insert(0, str(ROOT / "run"))
from pack import Ref  # noqa: E402
from train_enc import Enc  # noqa: E402
from common import options_of  # noqa: E402
from state import fit_account_state  # noqa: E402
M = ROOT / "web" / "models" / "account"
man = json.loads((M / "manifest.json").read_text()); data = (M / "weights.bin").read_bytes()
ref, tok = Ref(man, data), Enc(M / "tokenizer.json")
count = lambda t: len(tok.ids(t))
Q = json.loads((M / "questions.json").read_text()); PQ = json.loads((M / "post_question.json").read_text())
test = {json.loads(l)["account"] for l in (ROOT / "data" / "cases.jsonl").open() if '"split": "test"' in l}
accts = [a for a in (json.loads(l) for l in (ROOT / "data" / "accounts.jsonl").open()) if a["did"] in test][:8]
out = {"accounts": accts, "questions": Q, "post_question": PQ, "ref": []}
q = Q["account_type"]
for a in accts:
    st = fit_account_state(a, count)
    e = {"question": f"{q['type']}: {q['instructions']}", "options": list(options_of(q).items()), "state": st}
    ids, pos = tok.build(e)
    z = ref.logits(ids, pos) / man["temps"]["choice"]; z = np.exp(z - z.max()); out["ref"].append({"account_type": (z / z.sum()).tolist()})
(ROOT / "out" / "bridge").mkdir(parents=True, exist_ok=True)
(ROOT / "out" / "bridge" / "fixtures.json").write_text(json.dumps(out)); print("fixture accounts", len(accts))
