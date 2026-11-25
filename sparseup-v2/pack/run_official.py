"""The model repo's own code (trust_remote_code) on the parity fixture, for comparing with ref.py.

Run only inside claude-workspace scripts/jail.sh (no network, user nobody), with the hf/ files copied to model/:
    sh scripts/jail.sh -C <jail dir> -e HF_HUB_OFFLINE=1 -- python3 run_official.py model parity.json out/official.npy
Result: results/official_vs_ref.json (38/47 vectors bit-identical to ref.py, max |diff| 1.1e-5).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

torch.set_num_threads(2)
m = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, dtype=torch.float32, local_files_only=True).eval()
rows = json.loads(Path(sys.argv[2]).read_text())
out, ids_ok = [], 0
with torch.inference_mode():
    for r in rows:
        pre = m.config.query_prefix if r["kind"] == "query" else m.config.document_prefix
        ids, attn, pool, _ = m._tokenize([r["text"]], pre, 512)
        ids_ok += ids[0].tolist() == r["ids"] and pool[0].tolist() == r["pool"]
        out.append(m(ids, attn, pool)[0].float().numpy())
np.save(sys.argv[3], np.stack(out))
print("ids+pool identical", ids_ok, "/", len(rows))
