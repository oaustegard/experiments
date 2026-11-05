"""Score fresh checkpoints at full precision and packed at +-63 and +-31 codes, on the GPU.

    python3 run/quant_levels.py NAME [NAME ...]      # out/ckpt/NAME/{best.pt,meta.json}

For each variant the checkpoint goes through pack.pack(), the exact bytes a browser file would hold, and
the dequantized weights are read back from those bytes (via pack.Ref) into the PyTorch model, so the
score is the packed file's own, computed on the GPU in seconds instead of the NumPy reference's minutes.
A parity check compares those PyTorch logits with Ref on a few questions. Temperatures are the
checkpoint's own, fitted at full precision on validation, as in the shipped build.
Writes results/quant_levels-NAME.json: score per variant, brotli bytes of the whole weights file and of
the matmul codes alone, parity.
"""
import json
import sys
from pathlib import Path

import brotli
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, WORKFLOWS, examples, load_cases, score_examples  # noqa: E402
from pack import Ref, pack  # noqa: E402
from train_enc import DEV, Decider, Enc, predict, probs_at  # noqa: E402

VARIANTS = ["fp32", "t4-w8L63", "t4-w8L31"]
LIN = {"Wqkv": "attn.Wqkv.weight", "Wo": "attn.Wo.weight", "Wi": "mlp.Wi.weight", "Wo_mlp": "mlp.Wo.weight"}


def packed_state(sd, man, data):
    """The checkpoint's state dict with every packed tensor replaced by what the file decodes to."""
    r = Ref(man, data)
    out = {k: torch.tensor(v) for k, v in sd.items()}
    for i, L in enumerate(man["layers"]):
        for n, key in LIN.items():
            w = L[n]
            W = r.t(w["f32_t"]) if "f32_t" in w else (r.t(w["q"], np.int8).astype(np.float32) * r.t(w["s"])).T
            out[f"enc.layers.{i}.{key}"] = torch.tensor(np.ascontiguousarray(W))
    out["enc.embeddings.tok_embeddings.weight"] = torch.tensor(r.table(np.arange(man["table"]["V"])))
    return out, r


def codes_bytes(man, data):
    parts = []
    for L in man["layers"]:
        for n in LIN:
            rec = L[n].get("q") or L[n]["f32_t"]
            parts.append(data[rec["offset"]: rec["offset"] + rec["length"]])
    return b"".join(parts)


def main():
    from transformers import AutoConfig
    te = examples([c for w in WORKFLOWS for c in load_cases(w, "test")])
    for name in sys.argv[1:]:
        ck = ROOT / "out" / "ckpt" / name
        meta = json.loads((ck / "meta.json").read_text())
        temps = meta["temps"]
        state = torch.load(ck / "best.pt", map_location="cpu")
        sd = {k: v.numpy() for k, v in state.items()}
        cfg = AutoConfig.from_pretrained(meta["init"]).to_dict()
        tok = Enc(Path(meta["init"]) / "tokenizer.json")
        pad_id = tok.tok.token_to_id("[PAD]")
        data_te = []
        for e in te:
            ids, pos = tok.build(e)
            data_te.append({"ids": ids, "pos": pos, "gold": e["gold"]})
        model = Decider(meta["init"]).to(DEV)
        res = {"name": name, "temps": temps, "variants": {}}
        for v in VARIANTS:
            man, data = pack(sd, cfg, v, temps)
            st, ref = packed_state(sd, man, data)
            model.load_state_dict(st)
            logits = predict(model, data_te, pad_id)
            probs = [probs_at([z], temps[e["type"]])[0] for z, e in zip(logits, te)]
            summ, _ = score_examples(te, probs)
            by = {}
            for t in ("noul", "choice", "score"):
                sel = [i for i, e in enumerate(te) if e["type"] == t]
                by[t] = score_examples([te[i] for i in sel], [probs[i] for i in sel])[0]["accuracy"]
            par = max(float(np.abs(ref.logits(d["ids"], d["pos"]) - z).max()) for d, z in zip(data_te[:6], logits[:6]))
            res["variants"][v] = {"test": summ, "by_type": by, "file_bytes": len(data),
                                  "file_brotli": len(brotli.compress(data, quality=11)),
                                  "matmul_brotli": len(brotli.compress(codes_bytes(man, data), quality=11)),
                                  "parity_max_abs_logit_vs_ref": par,
                                  "top": [int(np.argmax(p)) for p in probs]}
            print(name, v, json.dumps({k: x for k, x in res["variants"][v].items() if k != "top"}), flush=True)
        for v in VARIANTS[1:]:
            a, b = res["variants"]["fp32"]["top"], res["variants"][v]["top"]
            res["variants"][v]["top_agree_with_fp32"] = float(np.mean([x == y for x, y in zip(a, b)]))
        for v in VARIANTS:
            res["variants"][v].pop("top")
        (ROOT / "results" / f"quant_levels-{name}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
