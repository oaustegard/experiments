"""System v0.1.0: pack the frozen base, the four workflow spikes and the router spike for the page.

    python3 -I run/pack_system.py [--score N] [--fixtures]

Writes web/models/:
  base/{weights.bin, manifest.json, tokenizer.json}  stock Ettin-17m in browser-decider's t4-w8L63 format
  spikes/<id>.bin + spikes/<id>.json                   one per spike, plus router.bin/json: rank-8 LoRA
      A and B (fp16, scale folded into B) for every Wqkv/Wo/Wi/Wo_mlp, and the decision head (fp16)
  system.json                                          spikes, router, threshold, version
  samples.json                                         demo cases (test split) and seed memories (train)

`LoraRef` (NumPy, from the packed files only) is the spec the page follows: --score N scores N test
cases per workflow through it, to confirm packing keeps each spike's accuracy; --fixtures writes
reference logits and pooled keys the page's check compares against.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import RANK, ALPHA, stock_path  # noqa: E402
from shared import BD, ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from pack import Packer, Ref, gelu, pack  # noqa: E402
from train_enc import Decider, Enc  # noqa: E402

VERSION = "0.1.0"
GAP_THRESHOLD = 0.5
WEB = ROOT / "web" / "models"
MODS = [("attn.Wqkv", "Wqkv"), ("attn.Wo", "Wo"), ("mlp.Wi", "Wi"), ("mlp.Wo", "Wo_mlp")]


def pack_spike(st, n_layers):
    """LoRA + head -> (manifest part, bytes). B carries the alpha/rank scale."""
    P = Packer()
    scale = ALPHA / RANK
    layers = []
    for i in range(n_layers):
        L = {}
        for mod, key in MODS:
            A = st[f"layers.{i}.{mod}.A"].float().numpy()
            B = st[f"layers.{i}.{mod}.B"].float().numpy() * scale
            L[key] = {"A": P.add(A, np.float16), "B": P.add(B, np.float16), "r": A.shape[0], "K": A.shape[1], "N": B.shape[0]}
        layers.append(L)
    h = {k[5:]: v.float().numpy() for k, v in st.items() if k.startswith("head.")}
    head = {"w0": P.add(h["0.weight"], np.float16), "b0": P.add(h["0.bias"], np.float16),
            "ln_w": P.add(h["2.weight"], np.float16), "ln_b": P.add(h["2.bias"], np.float16),
            "w1": P.add(h["3.weight"].reshape(-1), np.float16), "b1": float(h["3.bias"][0])}
    return {"layers": layers, "head": head}, P.bytes()


class LoraRef(Ref):
    """browser-decider's Ref plus a LoRA side path and a swappable head, read from a spike file."""

    def use(self, sm, sbytes):
        self.sm, self.sb = sm, sbytes

    def st(self, r):
        return np.frombuffer(self.sb, dtype=np.float16, count=r["length"] // 2, offset=r["offset"]).reshape(r["dims"]).astype(np.float32)

    def hidden(self, ids):
        self._li = -1
        return super().hidden(ids)

    def lin(self, x, w):
        y = super().lin(x, w)
        key = self._key(w)
        if key is None:
            return y
        L = self.sm["layers"][key[0]][key[1]]
        return y + (x @ self.st(L["A"]).T) @ self.st(L["B"]).T

    def _key(self, w):
        for i, L in enumerate(self.m["layers"]):
            for k in ("Wqkv", "Wo", "Wi", "Wo_mlp"):
                if L[k] is w:
                    return i, k
        return None

    def run(self, ids, pos, pool):
        hs = self.hidden(np.array(ids))
        hd = self.sm["head"]
        h = gelu(hs[pos] @ self.st(hd["w0"]).T + self.st(hd["b0"]))
        h = self.ln(h, self.st(hd["ln_w"]), self.st(hd["ln_b"]))
        key = hs[pool[0]: pool[1]].mean(0)
        return h @ self.st(hd["w1"]) + hd["b1"], key / (np.linalg.norm(key) + 1e-12)


def questions_of(w):
    import pandas as pd
    return json.loads(pd.read_parquet(BD / "data" / "td" / w / "test.parquet").questions.iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=int, default=0, help="score N test cases per workflow through LoraRef")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--smoke-router", action="store_true", help="use out/smoke's 2-step router (engine debugging only)")
    a = ap.parse_args()
    from transformers import AutoConfig
    init = stock_path()
    cfg = AutoConfig.from_pretrained(init).to_dict()
    sd = {k: v.numpy() for k, v in Decider(init).state_dict().items()}   # head is a placeholder: spikes carry their own
    man, data = pack(sd, cfg, "t4-w8L63", {})
    del man["head"]
    (WEB / "base").mkdir(parents=True, exist_ok=True)
    (WEB / "base" / "weights.bin").write_bytes(data)
    (WEB / "base" / "manifest.json").write_text(json.dumps(man))
    (WEB / "base" / "tokenizer.json").write_bytes((Path(init) / "tokenizer.json").read_bytes())
    print(f"base: {len(data) / 1e6:.2f} MB")

    rdir = ROOT / "out" / "smoke" if a.smoke_router else ROOT
    router = json.loads((rdir / ("system_router.json" if a.smoke_router else "results/system_router.json")).read_text())
    (WEB / "spikes").mkdir(exist_ok=True)
    system = {"version": VERSION, "rank": RANK, "router": {"file": "spikes/router", "question": router["question"],
              "none": router["none_desc"], "threshold": GAP_THRESHOLD,
              "threshold_note": "0.5, chosen after training (results/system_router_threshold.json): the 99th-percentile "
                                "rule gave 5.2e-5 and 3.5% false gaps; every in-library P(none) was below 6e-5 and every unseen one above 0.999"},
              "spikes": []}
    packed = {}
    for sid in ["router"] + WORKFLOWS:
        st = torch.load((rdir / "out" / "adapters" / "router-v1.pt" if not a.smoke_router else rdir / "adapters" / "router-v1.pt")
                        if sid == "router" else ROOT / "out" / "adapters" / f"S-lora-{sid}-s0.pt")
        sm, sb = pack_spike(st, cfg["num_hidden_layers"])
        if sid != "router":
            temps = json.loads((ROOT / "results" / f"S-lora-{sid}-s0.json").read_text())["temps"]
            sm.update({"id": sid, "description": router["spikes"][sid], "questions": questions_of(sid), "temps": temps})
            system["spikes"].append({"id": sid, "file": f"spikes/{sid}", "description": router["spikes"][sid]})
        (WEB / "spikes" / f"{sid}.bin").write_bytes(sb)
        (WEB / "spikes" / f"{sid}.json").write_text(json.dumps(sm))
        packed[sid] = (sm, sb)
        print(f"spike {sid}: {len(sb) / 1e3:.0f} kB")
    (WEB / "system.json").write_text(json.dumps(system, indent=1))

    # demo cases: 6 test cases per workflow with gold, 16 train cases per workflow as seed memory, 3 off-domain
    from system_router import synthetic
    samples = {"cases": [], "seed": {}, "offdomain": [
        {"label": "Lab sample record (a domain the router never saw)", "state": json.dumps(synthetic("lab", 3))},
        {"label": "Bank customer message (plain text)", "state": "I was charged twice for the same card payment yesterday."},
        {"label": "Restaurant booking (trained as none)", "state": json.dumps(synthetic("booking", 99))}]}
    for w in WORKFLOWS:
        cs = load_cases(w, "test")
        for c in random.Random(f"demo-{w}").sample(cs, 6):
            samples["cases"].append({"id": c["id"], "workflow": w, "state": c["state"],
                                     "gold": {q: g["label"] for q, g in c["gold"].items()}})
        tr = random.Random(f"seed-{w}").sample(load_cases(w, "train"), 16)
        samples["seed"][w] = [{"id": c["id"], "state": c["state"], "gold": {q: g["label"] for q, g in c["gold"].items()}} for c in tr]
    (WEB / "samples.json").write_text(json.dumps(samples))

    if a.score or a.fixtures:
        tok = Enc(Path(init) / "tokenizer.json")
        ref = LoraRef(man, data)
        fx = {"version": VERSION, "cases": []}
        for w in WORKFLOWS:
            ref.use(*packed[w])
            cases = load_cases(w, "test")[: a.score or 2]
            te = examples(cases)
            probs = []
            for e in te:
                ids, pos = tok.build(e)
                start = ids.index(tok.sep, pos[-1]) + 1
                z, key = ref.run(ids, pos, (start, len(ids) - 1))
                T = packed[w][0]["temps"][e["type"]]
                p = np.exp(z / T - (z / T).max())
                probs.append((p / p.sum()).tolist())
                if a.fixtures and len(fx["cases"]) < 8 and e["qname"] == te[0]["qname"]:
                    fx["cases"].append({"spike": w, "ids": ids, "pos": pos, "pool": [start, len(ids) - 1],
                                        "logits": z.tolist(), "key8": key[:8].tolist()})
            if a.score:
                summ = score_examples(te, probs)[0]
                print(f"LoraRef {w} on {len(cases)} test cases: acc {summ['accuracy']:.4f}", flush=True)
                fx[f"score_{w}"] = summ
        if a.fixtures:
            (WEB / "fixtures.json").write_text(json.dumps(fx))
        if a.score:
            (ROOT / "results" / "system_pack_check.json").write_text(json.dumps({k: v for k, v in fx.items() if k.startswith("score_")}, indent=1))


if __name__ == "__main__":
    main()
