"""System v0.1.0: compare gap thresholds for the trained router (after training; reported as a post hoc choice).

    python3 -I run/router_threshold.py
"""
import json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lora
from lora import load_adapter, new_model, predict, prep_raw, stock_path
from shared import BD, ROOT, WORKFLOWS, load_cases
from system_router import SPIKES, TEST_DOMAIN, example, synthetic
from train_enc import Enc, probs_at

torch.set_num_threads(4)
tok = Enc(Path(stock_path()) / "tokenizer.json"); lora.PAD = tok.tok.token_to_id("[PAD]")
model = new_model("lora"); load_adapter(model, torch.load(ROOT / "out/adapters/router-v1.pt")); model.eval()
texts = [json.loads(l)["input"] for l in open(BD / "data" / "decider_eval.jsonl")]; random.Random("none-texts").shuffle(texts)
sets = {"in_library": [(c["state"], w) for w in WORKFLOWS for c in load_cases(w, "test")],
        "unseen_texts": [(t, "none") for t in texts[70:]], "unseen_lab": [(json.dumps(synthetic(TEST_DOMAIN, i)), "none") for i in range(40)]}
P = {k: np.array([probs_at([z], 1.0)[0] for z in predict(model, prep_raw(tok, [example(s, l) for s, l in v]), lora.PAD)]) for k, v in sets.items()}
names = list(SPIKES) + ["none"]
out = {}
for thr in (json.loads((ROOT / "results/system_router.json").read_text())["threshold_p_none"], 0.5):
    route = lambda p: names[int(np.argmax(p[:-1]))] if p[-1] < thr else "none"
    out[f"{thr:.6g}"] = {"routing_acc": float(np.mean([route(p) == l for p, (_, l) in zip(P["in_library"], sets["in_library"])])),
                         "false_gap_rate": float(np.mean([route(p) == "none" for p in P["in_library"]])),
                         "gap_recall_texts": float(np.mean([route(p) == "none" for p in P["unseen_texts"]])),
                         "gap_recall_lab": float(np.mean([route(p) == "none" for p in P["unseen_lab"]]))}
out["max_p_none_in_library"] = float(P["in_library"][:, -1].max()); out["min_p_none_unseen"] = float(min(P["unseen_texts"][:, -1].min(), P["unseen_lab"][:, -1].min()))
out["routing_acc_ignoring_none"] = float(np.mean([names[int(np.argmax(p[:-1]))] == l for p, (_, l) in zip(P["in_library"], sets["in_library"])]))
(ROOT / "results/system_router_threshold.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
