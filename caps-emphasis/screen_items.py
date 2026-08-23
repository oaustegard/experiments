"""Keep only items where the forbidden word is the model's top-1 continuation
under no directive, with log-prob above a floor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import caps_lib as C
from items import CANDIDATES

FLOOR = -2.0  # nats; p >= 0.135

def main():
    tok, m = C.load()
    prompts = [C.build_prompt(q, pre) for _, _, q, pre in CANDIDATES]
    targets = [" " + w for _, w, _, _ in CANDIDATES]
    lps = C.logprobs(prompts, targets)
    rows, kept = [], []
    for (iid, w, q, pre), lp, p in zip(CANDIDATES, lps, prompts):
        with torch.no_grad():
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
            top = m(ids).logits[0, -1].argmax().item()
        top_str = tok.decode([top])
        tgt_ids = tok.encode(" " + w, add_special_tokens=False)
        is_top = (top == tgt_ids[0])          # first target token must be top-1
        single = len(tgt_ids) == 1
        ok = is_top and lp >= FLOOR
        rows.append(dict(id=iid, word=w, logp=round(lp, 4), top1=top_str,
                         is_top1=is_top, single_token=single, kept=ok))
        print(f"{'KEEP' if ok else 'drop'}  {iid:14} {w:12} logp={lp:7.3f} "
              f"top1={top_str!r} {'1tok' if single else str(len(tgt_ids))+'tok'}")
        if ok:
            kept.append(dict(id=iid, word=w, question=q, prefix=pre,
                             base_logp=lp, single_token=single,
                             first_token=tok.decode([tgt_ids[0]])))
    C.dump(os.path.join(os.path.dirname(os.path.abspath(__file__)), "items_screened.json"),
           dict(floor=FLOOR, n_candidates=len(CANDIDATES), n_kept=len(kept),
                rows=rows, kept=kept))
    print(f"\nkept {len(kept)}/{len(CANDIDATES)}")

if __name__ == "__main__":
    main()
