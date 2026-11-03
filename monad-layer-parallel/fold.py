"""Fold adjacent layer pairs of a Monad checkpoint into single double-width layers (Layer Parallelism,
arXiv 2502.02790, in the form that is one standard Llama layer), with no training.

    python3 fold.py select --model A8p --out results/pairs_A8p.json     # single-pair KL, pick pair sets
    python3 fold.py fold --model A8p --pairs results/pairs_A8p.json --k 16 --out A8p_P16   # write HF dir
    python3 fold.py check --model A8p --folded A8p_P16 --pairs ... --k 16                # fold == wrapper

A pair (A, B) becomes one layer with 8 heads and a 1,536-wide MLP:
    q/k/v rows   = [W_A · diag(n_A) ; W_B · diag(n_B)]       n = that layer's input RMSNorm weight
    o_proj cols  = [O_A , O_B]                               so the two attention outputs are summed
    gate/up rows = [G_A · diag(m_A) ; G_B · diag(m_B)]       m = post-attention RMSNorm weight
    down cols    = [D_A , D_B]
and both RMSNorm weights become 1 (--norm fold). The default (--norm mean) keeps g = (n_A + n_B)/2 as the
merged norm weight and folds n_A/g, n_B/g instead: folding the raw weights, which span 16x, into matrices
that are then quantized to int4 in blocks of 32 inputs roughly tripled the int4 error (Log, round 2). Both attention blocks read x and both MLPs read
h = x + A_A(x) + A_B(x), exactly lp_probe.Pair. A layer left unpaired is padded with zero heads
and zero MLP units, which contribute exactly 0, so the whole model has one uniform config that
onnxruntime-genai's builder accepts.
Models live in ../monad-weight-edit/models (gitignored), next to the harness that exports and runs them.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MWE = HERE.parent / "monad-weight-edit"
sys.path.insert(0, str(MWE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from common import load, prompt  # noqa: E402
from layers import kl  # noqa: E402
from lp_probe import Pair, build  # noqa: E402
from run_eval import question  # noqa: E402

FX = {x["uid"]: x for x in json.loads((MWE / "data/fixture.json").read_text())["dev"]}


def calib(tok, run, n=12, L=320):
    """Same recipe as layers.calib, on the given model's own dev answers."""
    recs = [json.loads(l) for l in open(MWE / "runs" / run)]
    per = {}
    for r in recs:
        per.setdefault(r["task"], []).append(r)
    pick = []
    while len(pick) < n:
        for t in ("mmlu", "gsm8k", "ifeval"):
            if per.get(t):
                pick.append(per[t].pop(0))
    seqs = []
    for r in pick[:n]:
        p = tok.encode(prompt(question(FX[r["uid"]])), add_special_tokens=False)
        g = tok.encode(r["text"], add_special_tokens=False)
        seqs.append(((p + g)[:L], len(p)))
    return seqs


@torch.no_grad()
def logprobs(model, seqs):
    out = []
    for ids, p in seqs:
        lg = model(torch.tensor([ids])).logits[0, p - 1:-1]
        out.append(torch.log_softmax(lg.float(), -1))
    return torch.cat(out)


def select(a):
    tok, model = load(MWE / "models" / a.model)
    layers = list(model.model.layers)
    seqs = calib(tok, a.run)
    model.model.layers = build(layers, [])
    base = logprobs(model, seqs)
    single = {}
    for k in range(len(layers) - 1):
        model.model.layers = build(layers, [k])
        single[k] = kl(base, logprobs(model, seqs))
    order = sorted(single, key=single.get)
    chosen, sets = [], {}
    for k in order:
        if any(abs(k - c) < 2 for c in chosen):
            continue
        chosen.append(k)
        if len(chosen) in (8, 16, 24):
            model.model.layers = build(layers, chosen)
            sets[len(chosen)] = {"pairs": sorted(chosen), "tf_kl": kl(base, logprobs(model, seqs))}
            print(len(chosen), round(sets[len(chosen)]["tf_kl"], 4), flush=True)
        if len(chosen) == 24:
            break
    Path(HERE / a.out).write_text(json.dumps({"model": a.model, "calib": a.run, "single_pair_kl": single,
                                              "sets": sets}, indent=1))


def fold(a):
    sets = json.loads((HERE / a.pairs).read_text())["sets"]
    pairs = set(sets[str(a.k)]["pairs"])
    tok, model = load(MWE / "models" / a.model)
    src = model.model.layers
    cfg = model.config.to_dict()
    nL = len(src) - len(pairs)
    cfg.update(num_hidden_layers=nL, num_attention_heads=8, num_key_value_heads=8, head_dim=64,
               intermediate_size=1536, torch_dtype="float32")
    cfg.pop("dtype", None)
    new = LlamaForCausalLM(LlamaConfig(**cfg)).float()
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("model.layers.")}
    groups, l = [], 0
    while l < len(src):
        if l in pairs:
            groups.append((src[l], src[l + 1]))
            l += 2
        else:
            groups.append((src[l], None))
            l += 1
    assert len(groups) == nL
    with torch.no_grad():
        for i, (A, B) in enumerate(groups):
            def gmean(get):
                """Merged layer's norm weight: the mean of the pair's two (all positive in Monad), so the
                factors folded into the int4-quantized matrices stay near 1 (see Log, round 2)."""
                if a.norm == "fold":
                    return torch.ones(256)
                if B is None:
                    return get(A).float()
                return (get(A).float() + get(B).float()) / 2
            g_in = gmean(lambda L: L.input_layernorm.weight)
            g_post = gmean(lambda L: L.post_attention_layernorm.weight)

            def part(L):
                if L is None:
                    z = lambda *s: torch.zeros(*s)  # noqa: E731
                    return {"q": z(256, 256), "k": z(256, 256), "v": z(256, 256), "o": z(256, 256),
                            "g": z(768, 256), "u": z(768, 256), "d": z(256, 768)}
                n = (L.input_layernorm.weight.float() / g_in)[None, :]
                m = (L.post_attention_layernorm.weight.float() / g_post)[None, :]
                at, ml = L.self_attn, L.mlp
                return {"q": at.q_proj.weight.float() * n, "k": at.k_proj.weight.float() * n,
                        "v": at.v_proj.weight.float() * n, "o": at.o_proj.weight.float(),
                        "g": ml.gate_proj.weight.float() * m, "u": ml.up_proj.weight.float() * m,
                        "d": ml.down_proj.weight.float()}
            pa, pb = part(A), part(B)
            p = f"model.layers.{i}."
            for key, name in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
                              ("g", "mlp.gate_proj"), ("u", "mlp.up_proj")):
                sd[p + name + ".weight"] = torch.cat([pa[key], pb[key]], 0)
            sd[p + "self_attn.o_proj.weight"] = torch.cat([pa["o"], pb["o"]], 1)
            sd[p + "mlp.down_proj.weight"] = torch.cat([pa["d"], pb["d"]], 1)
            sd[p + "input_layernorm.weight"] = g_in
            sd[p + "post_attention_layernorm.weight"] = g_post
    sd = {k: v.float() for k, v in sd.items()}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected and all("lm_head" in m for m in missing), (missing, unexpected)
    new.tie_weights()
    out = MWE / "models" / a.out
    shutil.rmtree(out, ignore_errors=True)
    new.save_pretrained(out)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
              "chat_template.json"):
        shutil.copy(MWE / "models/monad" / f, out / f)
    (out / "fold.json").write_text(json.dumps({"source": a.model, "pairs": sorted(pairs), "layers": nL}))
    print(out, nL, "layers")


def check(a):
    """Folded model == lp_probe.Pair wrapper on the source model (fp32), on 4 calibration sequences."""
    sets = json.loads((HERE / a.pairs).read_text())["sets"]
    pairs = sets[str(a.k)]["pairs"]
    tok, src = load(MWE / "models" / a.model)
    src.model.layers = build(list(src.model.layers), pairs)
    _, fol = load(MWE / "models" / a.folded)
    seqs = calib(tok, a.run, n=4, L=200)
    mx = 0.0
    with torch.no_grad():
        for ids, _ in seqs:
            x = torch.tensor([ids])
            mx = max(mx, float((src(x).logits - fol(x).logits).abs().max()))
    print(json.dumps({"folded": a.folded, "max_abs_logit_diff_vs_wrapper": mx}))
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "fold", "check"])
    ap.add_argument("--model", default="A8p")
    ap.add_argument("--run", default="A8p__gpu_dev.jsonl")
    ap.add_argument("--pairs", default="results/pairs_A8p.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="results/pairs_A8p.json")
    ap.add_argument("--folded")
    ap.add_argument("--norm", choices=["mean", "fold"], default="mean",
                    help="mean: merged RMSNorm weight = mean of the pair's, ratios folded; fold: all folded, weight 1")
    a = ap.parse_args()
    torch.set_num_threads(4)
    {"select": select, "fold": fold, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
