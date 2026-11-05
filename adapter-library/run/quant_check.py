"""System v0.1.0: where does packing lose accuracy? Score one spike through LoraRef on an fp32 base vs the shipped
int8/int4 base, with the same fp16 LoRA file.   python3 -I run/quant_check.py WORKFLOW N"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_system import LoraRef, pack, pack_spike
from lora import stock_path
from shared import ROOT, examples, load_cases, score_examples
from train_enc import Decider, Enc
from transformers import AutoConfig
w, N = sys.argv[1], int(sys.argv[2])
init = stock_path(); cfg = AutoConfig.from_pretrained(init).to_dict()
sd = {k: v.numpy() for k, v in Decider(init).state_dict().items()}
st = torch.load(ROOT / "out/adapters" / f"S-lora-{w}-s0.pt"); sm, sb = pack_spike(st, cfg["num_hidden_layers"])
temps = json.loads((ROOT / "results" / f"S-lora-{w}-s0.json").read_text())["temps"]
tok = Enc(Path(init) / "tokenizer.json"); te = examples(load_cases(w, "test")[:N]); out = {}
for variant in ("fp32", "t4-w8L63"):
    man, data = pack(sd, cfg, variant, {}); ref = LoraRef(man, data); ref.use(sm, sb); probs = []
    for e in te:
        ids, pos = tok.build(e); start = ids.index(tok.sep, pos[-1]) + 1
        z, _ = ref.run(ids, pos, (start, len(ids) - 1)); z = z / temps[e["type"]]; p = np.exp(z - z.max()); probs.append((p / p.sum()).tolist())
    out[variant] = score_examples(te, probs)[0]["accuracy"]; print(w, variant, out[variant], flush=True)
(ROOT / "results" / f"quant_check_{w}.json").write_text(json.dumps({"n_cases": N, **out}, indent=1))
