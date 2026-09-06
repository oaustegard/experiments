"""Side-by-side demo: frozen model vs text tool vs latent (stream) port.

    python3 demo.py --model smol --n 12 [--seed 0] [--split test_in|test_len5]
                    [--head attn] [--prompts "12 + 3 =" ...] [--json out.json]

Three arms per prompt, one loaded model:

  none    the frozen model answering on its own;
  text    the classic tool call -- the exact result spliced into the prompt as
          tokens (oracle result when the row's operands are known);
  stream  the latent port -- the result encoded into one vector per answer
          step and added to the residual stream after layer k, with the query
          read out of the prompt by the learned query head (`--head attn` by
          default here) and run through the python calculator.  No tokens
          cross the boundary in either direction.

Arms whose checkpoint is missing are skipped with a line saying so, so the
demo runs against whatever has been trained.
"""

import argparse
import json
import os
import time

import torch

import data as D
import eval as E
import model_utils as mu
import query_head as QH

ARMS = ["none", "text", "stream"]


def ckpt(model_name, arm):
    return os.path.join(mu.repo_dir(), "ckpt", f"{model_name}_{arm}_final.pt")


def pick_rows(args):
    if args.prompts:
        return [{"prompt": p} for p in args.prompts]
    rows = D.load_split(args.split)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(rows), generator=g)[:args.n].tolist()
    return [rows[i] for i in idx]


def decode_query(op_logits, slot_logits):
    """(pretty 'op a b' string, calculator result string) for one row."""
    op = int(op_logits.argmax(-1))
    slots = slot_logits.argmax(-1).tolist()
    a = mu.decode_operand(slots[:D.N_OPERAND_SLOTS])
    b = mu.decode_operand(slots[D.N_OPERAND_SLOTS:])
    return f"{D.OPS[op]} {a} {b}", mu.calculate(op, slots[:D.N_OPERAND_SLOTS],
                                                slots[D.N_OPERAND_SLOTS:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test_in",
                    choices=["test_in", "test_len5", "val"])
    ap.add_argument("--head", default="attn", choices=["mlp", "attn"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--max-new", type=int, default=E.MAX_NEW)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.k is None:
        with open(os.path.join(mu.repo_dir(), "results",
                               f"probe_{args.model}.json")) as f:
            args.k = json.load(f)["chosen_k"]
    k = args.k

    rows = pick_rows(args)
    model, tok = mu.load_model(args.model)
    hook = mu.attach_hook(model, k)

    arms = ["none", "text"]
    skipped = {}
    enc = None
    if os.path.exists(ckpt(args.model, "stream")):
        enc = mu.ResultEncoder(mu.hidden_size(model), n_steps=mu.N_STREAM_STEPS)
        enc.load_state_dict(torch.load(ckpt(args.model, "stream"))["state_dict"])
        enc.eval()
    else:
        skipped["stream"] = f"missing {ckpt(args.model, 'stream')}"

    qh = None
    qh_path = QH.ckpt_file(args.model, args.head)
    if os.path.exists(qh_path):
        qh = E.load_query_head(model, args.model, args.head)
    else:
        skipped["query_head"] = f"missing {qh_path}"
    if enc is not None and qh is not None:
        arms.append("stream")
    elif enc is not None:
        skipped["stream"] = skipped.get("query_head", "no query head")

    for name, why in skipped.items():
        print(f"skip {name}: {why}", flush=True)

    print(f"model={args.model} k={k} head={args.head} arms={' '.join(arms)}\n",
          flush=True)

    recs = []
    for i, r in enumerate(rows):
        prompt = r["prompt"]
        gold = r.get("result_string")
        rec = {"prompt": prompt, "gold": gold, "arms": {}}
        print(f"[{i + 1}] {prompt}")
        print(f"    gold        : {gold if gold is not None else '(unknown)'}")

        # --- latent query readout (shared by the stream arm, printed once)
        query_txt = calc_res = None
        if qh is not None:
            with torch.no_grad():
                op, slot = E.query_logits(model, tok, [prompt], k, qh[0], qh[1],
                                          args.head)
            query_txt, calc_res = decode_query(op[0], slot[0])
            rec["query"] = query_txt
            rec["calculator"] = calc_res

        for arm in arms:
            t0 = time.time()
            p, tool_tok, vec, syms = prompt, 0, None, None
            if arm == "text":
                res = (D.compute(r["op"], r["a"], r["b"])[0] if gold is not None
                       else calc_res)
                if res is None:
                    continue
                p = E.text_prompt(prompt, res)
                tool_tok = len(tok(" [" + res + "]",
                                   add_special_tokens=False)["input_ids"])
            elif arm == "stream":
                syms = mu.result_symbols([calc_res])
            gen, ntok, _ = E.generate(model, tok, [p], arm, k, vec, bs=1,
                                      max_new=args.max_new, enc=enc, syms=syms)
            ms = (time.time() - t0) * 1000
            n_prompt = len(tok(p)["input_ids"])
            ok = None if gold is None else int(gen[0] == gold)
            tot = n_prompt + tool_tok + ntok[0]
            rec["arms"][arm] = {"gen": gen[0], "correct": ok, "tokens": tot,
                                "ms": ms, "prompt_tokens": n_prompt,
                                "tool_tokens": tool_tok,
                                "generated_tokens": ntok[0]}
            mark = "?" if ok is None else ("OK " if ok else "X  ")
            extra = ""
            if arm == "text":
                extra = f'  tool="{" [" + res + "]"}"'
            elif arm == "stream":
                extra = f'  query="{query_txt}" -> calc={calc_res}'
            print(f"    {arm:<11} : {mark} {gen[0]!r}{extra}")
            print(f"    {'':<11}   tokens {tot} ({n_prompt}p+{tool_tok}t"
                  f"+{ntok[0]}g)  {ms:.0f} ms")
        recs.append(rec)
        print(flush=True)

    summary = {}
    for arm in arms:
        got = [r["arms"][arm] for r in recs if arm in r["arms"]]
        if not got:
            continue
        corr = [a["correct"] for a in got if a["correct"] is not None]
        summary[arm] = {
            "n": len(got),
            "accuracy": (sum(corr) / len(corr)) if corr else None,
            "mean_tokens": sum(a["tokens"] for a in got) / len(got),
            "mean_ms": sum(a["ms"] for a in got) / len(got)}

    print(f"{'arm':<11} {'n':>3} {'acc':>6} {'tokens':>8} {'ms':>8}")
    for arm, sm in summary.items():
        acc = "  n/a" if sm["accuracy"] is None else f"{sm['accuracy']:.3f}"
        print(f"{arm:<11} {sm['n']:>3} {acc:>6} {sm['mean_tokens']:>8.1f} "
              f"{sm['mean_ms']:>8.0f}")

    hook.remove()
    if args.json:
        out = {"model": args.model, "k": k, "head": args.head,
               "split": None if args.prompts else args.split,
               "n": len(recs), "seed": args.seed, "skipped": skipped,
               "summary": summary, "rows": recs}
        mu.ensure_dir(os.path.dirname(os.path.abspath(args.json)))
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"-> {args.json}")
    print("DONE demo")


if __name__ == "__main__":
    main()
