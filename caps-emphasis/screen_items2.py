"""Screen items into baseline-pressure strata.

Rana (arXiv:2601.08070) shows violation probability is a logistic function of the
model's intrinsic probability of the forbidden token, so an unstratified item
pool reports a property of the pool. This bins items by that baseline pressure
and picks a control token per item at the same time.
"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import caps_lib as C
from items import CANDIDATES

STRATA = [("low", -5.0, -2.0), ("mid", -2.0, -0.8),
          ("high", -0.8, -0.25), ("ceiling", -0.25, 0.0)]
BAD_CONTROL = set(' ,.:;?!"\'()[]-\n')


def main():
    tok, m = C.load()
    kept = []
    for iid, w, q, pre in CANDIDATES:
        p = C.build_prompt(q, pre)
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            lp = torch.log_softmax(m(ids).logits[0, -1].float(), -1)
        tgt = tok.encode(" " + w, add_special_tokens=False)
        base = lp[tgt[0]].item()
        rank = int((lp > lp[tgt[0]]).sum().item())
        if base < -5.0 or rank > 9:
            print(f"drop {iid:14} base={base:6.2f} rank={rank}")
            continue
        # control: highest-probability alternative that is a real word
        ctrl = None
        for cid in lp.topk(30).indices.tolist():
            if cid == tgt[0]:
                continue
            s = tok.decode([cid])
            if len(s.strip()) >= 3 and not set(s.strip()) & BAD_CONTROL:
                ctrl = s
                break
        if ctrl is None:
            print(f"drop {iid:14} no usable control token")
            continue
        stratum = next((n for n, lo, hi in STRATA if lo <= base < hi), "ceiling")
        kept.append(dict(id=iid, word=w, question=q, prefix=pre, base_logp=base,
                         base_rank=rank, control=ctrl, stratum=stratum,
                         control_logp=lp[tok.encode(ctrl, add_special_tokens=False)[0]].item()))
        print(f"KEEP {iid:14} base={base:6.2f} rank={rank} stratum={stratum:8} "
              f"control={ctrl!r}")
    counts = {n: sum(1 for k in kept if k["stratum"] == n) for n, _, _ in STRATA}
    C.dump(os.path.join(HERE, "items_v2.json"),
           dict(n=len(kept), strata=counts, items=kept))
    print("\nstrata:", counts, " total", len(kept))


if __name__ == "__main__":
    main()
