"""A ±31 base for the Spiky Ball: pack stock Ettin-17m with matmul codes in ±31 and score the shipped spikes on it.

    python3 -I run/base31.py [--cases N]      # N test cases per spike (default: all)

browser-decider found ±31 per-column codes cost about 0.002 against ±63 and save 0.5 MB, on specialists
packed from fp32. The Spiky Ball's base is stock Ettin-17m, and its spikes were fine-tuned against the ±63
base, so this checks the shipped spike files (web/models/<set>/spikes) on both bases through LoraRef, the
NumPy spec the page follows. It first repacks ±63 and requires it to match the shipped weights.bin byte for
byte, so both bases come from the same stock weights. Writes out/base31/{weights.bin, manifest.json} and
results/base31.json: per spike, accuracy and KL on both bases; for the router, the share of test cases routed
to their own spike.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import stock_path  # noqa: E402
from pack_system import WEB, LoraRef, everyday_cases  # noqa: E402
from shared import BD, ROOT, examples, load_cases, score_examples  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from pack import pack  # noqa: E402
from train_enc import Decider, Enc  # noqa: E402


def base(sd, cfg, variant):
    man, data = pack(sd, cfg, variant, {})
    data = data[: man["head"]["w0"]["offset"]]   # as pack_system.py: spikes carry their own heads
    del man["head"]
    return man, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0)
    a = ap.parse_args()
    from transformers import AutoConfig
    init = stock_path()
    cfg = AutoConfig.from_pretrained(init).to_dict()
    sd = {k: v.numpy() for k, v in Decider(init).state_dict().items()}
    m63, d63 = base(sd, cfg, "t4-w8L63")
    shipped = (WEB / "base" / "weights.bin").read_bytes()
    same = d63 == shipped and m63 == json.loads((WEB / "base" / "manifest.json").read_text())
    print(f"±63 repack identical to the shipped base: {same}", flush=True)
    if not same:
        sys.exit("the ±63 repack does not reproduce the shipped base; the comparison would not be like for like")
    m31, d31 = base(sd, cfg, "t4-w8L31")
    out = ROOT / "out" / "base31"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(d31)
    (out / "manifest.json").write_text(json.dumps(m31))
    import brotli
    sizes = {v: {"raw_mb": len(d) / 1e6, "brotli_mb": len(brotli.compress(d, quality=11)) / 1e6} for v, d in (("L63", d63), ("L31", d31))}
    print("sizes", sizes, flush=True)
    refs = {"L63": LoraRef(m63, d63), "L31": LoraRef(m31, d31)}
    tok = Enc(Path(init) / "tokenizer.json")
    res = {"sizes": sizes, "sha256_L31": hashlib.sha256(d31).hexdigest(), "sets": {}}
    for set_ in ("workflows", "everyday"):
        system = json.loads((WEB / set_ / "system.json").read_text())
        spikes = {s["id"]: s for s in system["spikes"]}
        load = lambda f: (json.loads((WEB / set_ / f"{f}.json").read_text()), (WEB / set_ / f"{f}.bin").read_bytes())
        rs = res["sets"][set_] = {"spikes": {}, "router": {}}
        cases_of = (lambda w: load_cases(w, "test")) if set_ == "workflows" else (lambda w: everyday_cases(w, "test"))
        all_cases = {w: cases_of(w)[: a.cases or None] for w in spikes}
        for w in spikes:
            sm, sb = load(spikes[w]["file"])
            te = examples(all_cases[w])
            for v, ref in refs.items():
                ref.use(sm, sb)
                probs = []
                for e in te:
                    ids, pos = tok.build(e)
                    start = ids.index(tok.sep, pos[-1]) + 1
                    z, _ = ref.run(ids, pos, (start, len(ids) - 1))
                    T = sm["temps"].get(e["type"], 1.0)
                    p = np.exp(z / T - (z / T).max())
                    probs.append((p / p.sum()).tolist())
                s = score_examples(te, probs)[0]
                rs["spikes"].setdefault(w, {})[v] = {"n": len(te), "accuracy": s["accuracy"], "kl": s["kl_from_gold"]}
                print(f"{set_} {w} {v}: acc {s['accuracy']:.4f} kl {s['kl_from_gold']:.4f} ({len(te)} decisions)", flush=True)
        rm, rb = load(system["router"]["file"])
        opts = [(s["id"], s["description"]) for s in system["spikes"]] + [("none", system["router"]["none"])]
        for v, ref in refs.items():
            ref.use(rm, rb)
            ok = n = 0
            for w, cs in all_cases.items():
                for c in cs:
                    e = {"question": system["router"]["question"], "options": opts, "state": c["state"]}
                    ids, pos = tok.build(e)
                    start = ids.index(tok.sep, pos[-1]) + 1
                    z, _ = ref.run(ids, pos, (start, len(ids) - 1))
                    p = np.exp(z - z.max()); p /= p.sum()
                    ok += opts[int(np.argmax(p))][0] == w and p[-1] < system["router"]["threshold"]
                    n += 1
            rs["router"][v] = {"n": n, "routed_to_own_spike": ok / n}
            print(f"{set_} router {v}: {ok}/{n}", flush=True)
    (ROOT / "results" / "base31.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
