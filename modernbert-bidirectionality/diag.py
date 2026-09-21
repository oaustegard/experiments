"""Post-hoc diagnostic arms (added after the main run, so not pre-registered):
is the layer-0 dependence the [CLS] query row building the sink?

  diag:l0-clsrow          single:0 (layer 0 causal) but query row 0 ([CLS]) reads everything at layer 0
  diag:l0-clsrow+special  the same, plus every query may read [CLS]/[SEP] at layer 0
  diag:causal+clsrow+special  every layer causal + sinks readable, and row 0 open at every layer
Writes results/diag_<tag>.json. Completion line: 'done -> <path>'."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import masks, probe

def main():
    model_name, tag = sys.argv[1], sys.argv[2]
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="eager").eval()
    lt = list(model.config.layer_types); N = len(lt)
    ids = probe.load_corpus_ids(256); ids_m, labels, masked_bool = probe.build_masked_inputs(ids)
    ctx = probe.build_ctx(lt, model.config.sliding_window, ids, tok, masked_bool)
    base = json.load(open(HERE / f"results/{tag}.json"))["arms"]
    ce_b, ce_t = base["bidir"]["ce"], base["truncate"]["ce"]
    def arm_mask(name):
        if name == "diag:l0-clsrow":
            a = masks.allowed("single:0", None, ctx); a[0, 0, :] = ctx["base"][0, 0, :]
        elif name == "diag:l0-clsrow+special":
            a = masks.allowed("single+special:0", None, ctx); a[0, 0, :] = ctx["base"][0, 0, :]
        elif name == "diag:causal+clsrow+special":
            a = masks.allowed("causal+special", None, ctx)
            for l in range(N): a[l, 0, :] = ctx["base"][l, 0, :]
        return a
    out = {}
    holder, handles = masks.install_mask_hooks(model.model.layers)
    dtype = next(model.parameters()).dtype
    for name in ("diag:l0-clsrow", "diag:l0-clsrow+special", "diag:causal+clsrow+special"):
        add = masks.to_additive(arm_mask(name), dtype).unsqueeze(0)
        tl = tc = tn = 0; t0 = time.time()
        with torch.no_grad():
            for s in range(0, 256, 8):
                holder.mask = add
                x = torch.from_numpy(ids_m[s:s+8]); y = torch.from_numpy(labels[s:s+8])
                lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
                mb = y != -100; fl = lg[mb].float(); g = y[mb]
                tl += torch.nn.functional.cross_entropy(fl, g, reduction="sum").item(); tc += (fl.argmax(-1) == g).sum().item(); tn += g.numel()
        ce = tl / tn
        out[name] = {"ce": ce, "acc": tc / tn, "n_masked": tn, "seconds": time.time() - t0,
                     "delta_ce": ce - ce_b, "retained_trunc": (ce_t - ce) / (ce_t - ce_b)}
        print(f"arm={name} ce={ce:.4f} acc={tc/tn:.3f} retained_trunc={out[name]['retained_trunc']:+.3f} s={time.time()-t0:.1f}", flush=True)
    for h in handles: h.remove()
    # budget bookkeeping for the anchor arms: mean set sizes per sequence
    out["_anchor_set_sizes"] = {"punct": float(np.mean([int(p.sum()) for p in ctx["punct"]])),
                                "masked": float(np.mean([int(m.sum()) for m in ctx["masked"]])),
                                "attn:5": probe.k_for_p(5), "attn:10": probe.k_for_p(10), "attn:25": probe.k_for_p(25)}
    print("anchor set sizes", out["_anchor_set_sizes"])
    p = HERE / f"results/diag_{tag}.json"; json.dump(out, open(p, "w"), indent=1); print(f"done -> {p.relative_to(HERE)}")

if __name__ == "__main__":
    main()
