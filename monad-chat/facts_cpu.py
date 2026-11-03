"""Greedy answers to probes_facts.json from the shipped 8-bit export on onnxruntime CPU (ort_gen.py's loop).
    python3 facts_cpu.py A8p_q8 runs/facts_cpu.jsonl
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "monad-weight-edit"))
from ort_gen import session, generate, pieces  # noqa: E402
from bridge import tok  # noqa: E402
from common import split_think  # noqa: E402
name, out = sys.argv[1], HERE / sys.argv[2]
t = tok(); P = pieces(t)
sess = session(str(HERE.parent / "monad-weight-edit/export" / name), threads=4)
probes = json.loads((HERE / "probes_facts.json").read_text())
ids = [t.encode(f"<|im_start|>user\n{p['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False) for p in probes]
with out.open("a") as f:
    for k in range(0, len(ids), 10):
        r = generate(sess, ids[k:k + 10], 768, P)
        toks = r["tokens"] if isinstance(r, dict) else r[0]
        for p, tk in zip(probes[k:k + 10], toks):
            _, ans, _ = split_think(t.decode(tk))
            f.write(json.dumps({"model": name, "probe": p["id"], "q": p["q"], "ok": bool(re.search(p["check"], ans, re.I)), "answer": ans}) + "\n")
        print(k, flush=True)
