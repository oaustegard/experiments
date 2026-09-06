"""Greedy-decode evaluation of every arm x query mode.

Generation uses LEFT padding so every row's prompt ends at the same absolute
index (t = T-1), which keeps the injection index and the latent-slot mask
uniform across a batch.  The prompt is run once with the injection active and
use_cache=True; the decode steps then run with the residual hook DISABLED (the
injected information already lives in the cached K/V of layers > k at position
t).  Exceptions: the `kv` slot stays active for every decode step (all decode
query positions are >= t, so it stays visible), and the `delayed` arm injects
during the FIRST decode step, which is where position t+1 is computed.
"""

import argparse
import json
import os
import time

import torch

import data as D
import model_utils as mu
import query_head as QH

MAX_NEW = 16


def set_ctx(mode=None, k=None, vec=None, t=None, q_offset=0, ek=None, ev=None):
    ctx = mu.get_ctx()
    ctx.enabled = mode is not None
    ctx.mode, ctx.layer_k, ctx.vec, ctx.t = mode, k, vec, t
    ctx.q_offset, ctx.extra_k, ctx.extra_v = q_offset, ek, ev
    return ctx


@torch.no_grad()
def generate(model, tok, prompts, arm, k, vec=None, max_new=MAX_NEW, bs=16,
             stop=True, enc=None, syms=None):
    """Returns (strings, generated-token counts, generated token ids).

    The `stream` arm needs the encoder itself (`enc`) and the result symbols
    (`syms`) rather than one precomputed vector, because it injects a DIFFERENT
    vector at every step: j = -1 during the prompt pass (at position t) and
    j = step during decode step `step` (at position t+1+step, which is exactly
    the position that step computes).
    """
    outs, ntoks, all_ids = [], [], []
    newline = tok(  "\n", add_special_tokens=False)["input_ids"][-1]
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        rows = [{"prompt": p, "answer": ""} for p in chunk]
        b = mu.encode_rows(tok, rows, with_answer=False, pad_side="left")
        ids, mask = b["input_ids"], b["attention_mask"]
        t = b["t"]
        v = None if vec is None else vec[i:i + len(chunk)]
        sy = None if syms is None else syms[i:i + len(chunk)]
        ek = ev = None
        if arm == "kv":
            ek, ev = mu.make_slot_kv(model, k, v)
        pos = mu.position_ids_from_mask(mask)
        if arm in ("residual",):
            set_ctx("residual", k, v, t, q_offset=0)
        elif arm == "stream":
            set_ctx("stream", k, enc(sy, step=-1), t, q_offset=0)
        elif arm == "kv":
            set_ctx("kv", k, v, t, ek=ek, ev=ev)
        else:
            set_ctx(None)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                    past_key_values=None, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        done = torch.zeros(len(chunk), dtype=torch.bool)
        gen = [[] for _ in chunk]
        last_pos = pos[:, -1]
        for step in range(max_new):
            for j in range(len(chunk)):
                if not done[j]:
                    gen[j].append(int(nxt[j]))
                    # Stop at the end of the first NON-EMPTY line: the frozen
                    # base models like to open a continuation with "\n\n", and
                    # halting on that would score every untrained baseline as
                    # an empty answer regardless of what it goes on to say.
                    if stop and int(nxt[j]) in (newline, tok.eos_token_id):
                        if tok.decode(gen[j], skip_special_tokens=True).strip():
                            done[j] = True
            if bool(done.all()) or step == max_new - 1:
                break
            mask = torch.cat([mask, torch.ones(len(chunk), 1, dtype=mask.dtype)], 1)
            last_pos = last_pos + 1
            if arm == "delayed" and step == 0:
                set_ctx("delayed", k, v, t, q_offset=int(t[0]) + 1)
            elif arm == "stream":
                # this forward computes position t+1+step, i.e. answer step j=step
                tgt = t + 1 + step
                set_ctx("stream", k, enc(sy, step=step), tgt,
                        q_offset=int(t[0]) + 1 + step)
            elif arm == "kv":
                set_ctx("kv", k, v, t, ek=ek, ev=ev)
            else:
                set_ctx(None)
            out = model(input_ids=nxt[:, None], attention_mask=mask,
                        position_ids=last_pos[:, None], past_key_values=cache,
                        use_cache=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        set_ctx(None)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j], skip_special_tokens=True)
            outs.append(txt.strip().split("\n")[0].strip())
            ntoks.append(len(gen[j]))
            all_ids.append(list(gen[j]))
    return outs, ntoks, all_ids


def load_query_head(model, model_name, head="mlp"):
    ck = torch.load(QH.ckpt_file(model_name, head))
    mod = (mu.QueryHead(mu.hidden_size(model)) if head == "mlp"
           else mu.AttnQueryHead(mu.hidden_size(model)))
    mod.load_state_dict(ck["state_dict"])
    mod.eval()
    return mod, ck


def query_logits(model, tok, prompts, k, mod, ck, head="mlp"):
    with torch.no_grad():
        if head == "mlp":
            h = mu.extract_hidden(model, tok, prompts)
            return mod((h[:, k + 1, :].float() - ck["mean"]) / ck["std"])
        hs, mask = mu.extract_hidden_seq(model, tok, prompts, k)
        return mod(hs.float(), mask)


def learned_results(model, tok, rows, model_name, k, head="mlp", cache=None):
    """Query head argmax -> calculator, for every row."""
    if cache is None:
        cache = load_query_head(model, model_name, head)
    mod, ck = cache
    op, slot = query_logits(model, tok, [r["prompt"] for r in rows], k, mod, ck,
                            head)
    return mu.calculate_from_logits(op, slot)


def oracle_results(rows):
    return [D.compute(r["op"], r["a"], r["b"])[0] for r in rows]


def text_prompt(prompt, result):
    return prompt + " [" + result + "]"


def breakdown(rows, correct):
    by_op, by_len = {}, {}
    for r, c in zip(rows, correct):
        by_op.setdefault(r["op"], []).append(c)
        by_len.setdefault(max(r["lengths"]), []).append(c)
    def fmt(d):
        return {str(kk): {"n": len(v), "acc": sum(v) / len(v)}
                for kk, v in sorted(d.items(), key=lambda x: str(x[0]))}
    return fmt(by_op), fmt(by_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(mu.MODELS))
    ap.add_argument("--arm", required=True,
                    choices=["none", "text", "residual", "kv", "delayed",
                             "stream"])
    ap.add_argument("--query", required=True, choices=["oracle", "learned"])
    ap.add_argument("--head", default="mlp", choices=["mlp", "attn"],
                    help="query head used when --query learned")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--n-timing", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.k is None and args.arm != "none":
        with open(os.path.join(mu.repo_dir(), "results",
                               f"probe_{args.model}.json")) as f:
            args.k = json.load(f)["chosen_k"]
    k = args.k
    model, tok = mu.load_model(args.model)
    hook = None
    if args.arm in ("residual", "delayed", "stream"):
        hook = mu.attach_hook(model, k)
    enc = None
    if args.arm in ("residual", "kv", "delayed", "stream"):
        ck = torch.load(os.path.join(mu.repo_dir(), "ckpt",
                                     f"{args.model}_{args.arm}_final.pt"))
        enc = mu.ResultEncoder(
            mu.hidden_size(model),
            n_steps=mu.N_STREAM_STEPS if args.arm == "stream" else 0)
        enc.load_state_dict(ck["state_dict"])
        enc.eval()
    qh_cache = (load_query_head(model, args.model, args.head)
                if args.query == "learned" and args.arm != "none" else None)

    res = {"model": args.model, "arm": args.arm, "query": args.query,
           "head": args.head, "k": k, "n_eval": args.n_eval, "splits": {}}

    for split in ("test_in", "test_len5"):
        rows = D.load_split(split, args.n_eval)
        t_q0 = time.time()
        # the `none` arm consumes no result at all, so the query mode is
        # irrelevant there and we skip the (costly) query-head pass
        results = (oracle_results(rows)
                   if args.query == "oracle" or args.arm == "none"
                   else learned_results(model, tok, rows, args.model, k,
                                        args.head, qh_cache))
        query_wall = time.time() - t_q0
        prompts = [r["prompt"] for r in rows]
        tool_tokens = [0] * len(rows)
        vec, syms = None, None
        if args.arm == "text":
            prompts = [text_prompt(p, r) for p, r in zip(prompts, results)]
            tool_tokens = [
                len(tok(" [" + r + "]", add_special_tokens=False)["input_ids"])
                for r in results]
        elif args.arm == "stream":
            syms = mu.result_symbols(results)
        elif args.arm in ("residual", "kv", "delayed"):
            with torch.no_grad():
                vec = enc(mu.result_symbols(results))
        t0 = time.time()
        gen, ntok, _ = generate(model, tok, prompts, args.arm, k, vec,
                                bs=args.bs, enc=enc, syms=syms)
        wall = time.time() - t0
        correct = [1 if g == r["result_string"] else 0
                   for g, r in zip(gen, rows)]
        # secondary, laxer metric: the frozen baselines often wrap the right
        # number in prose ("Final answer: **5505**"), which exact match scores 0
        contains = [1 if r["result_string"] in g else 0
                    for g, r in zip(gen, rows)]
        by_op, by_len = breakdown(rows, correct)
        n_prompt = [len(tok(r["prompt"])["input_ids"]) for r in rows]
        res["splits"][split] = {
            "n": len(rows),
            "exact_match": sum(correct) / len(correct),
            "contains_match": sum(contains) / len(contains),
            "by_op": by_op, "by_max_len": by_len,
            "tokens_per_answer": (sum(n_prompt) + sum(tool_tokens)
                                  + sum(ntok)) / len(rows),
            "prompt_tokens": sum(n_prompt) / len(rows),
            "tool_tokens": sum(tool_tokens) / len(rows),
            "generated_tokens": sum(ntok) / len(rows),
            "batched_wall_seconds": wall,
            "query_wall_seconds": query_wall,
            "calculator_exact": sum(
                1 for r, q in zip(rows, results) if q == r["result_string"]
            ) / len(rows),
            "examples": [{"prompt": rows[i]["prompt"], "gold":
                          rows[i]["result_string"], "gen": gen[i]}
                         for i in range(min(5, len(rows)))],
        }
        print(split, "exact", res["splits"][split]["exact_match"], flush=True)

    # latency: batch 1, full per-row pipeline
    rows = D.load_split("test_in", args.n_timing)
    t0 = time.time()
    for r in rows:
        one = [r]
        rr = (oracle_results(one)
              if args.query == "oracle" or args.arm == "none"
              else learned_results(model, tok, one, args.model, k, args.head,
                                   qh_cache))
        p = [text_prompt(r["prompt"], rr[0])] if args.arm == "text" else [r["prompt"]]
        v, sy = None, None
        if args.arm == "stream":
            sy = mu.result_symbols(rr)
        elif enc is not None:
            with torch.no_grad():
                v = enc(mu.result_symbols(rr))
        generate(model, tok, p, args.arm, k, v, bs=1, enc=enc, syms=sy)
    res["cpu_ms_per_answer_bs1"] = (time.time() - t0) / len(rows) * 1000
    res["n_timing"] = len(rows)

    if hook is not None:
        hook.remove()
    suffix = QH.head_suffix(args.head) if args.query == "learned" else ""
    path = args.out or os.path.join(
        mu.repo_dir(), "results",
        f"{args.model}_{args.arm}_{args.query}{suffix}.json")
    mu.ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"ms/answer {res['cpu_ms_per_answer_bs1']:.1f} -> {path}")
    print(f"DONE eval {args.model} {args.arm} {args.query} {args.head}")


if __name__ == "__main__":
    main()
