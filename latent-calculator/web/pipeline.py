"""Python onnxruntime reference of the exact algorithm the browser runs.

    python3 pipeline.py --n 20 [--variant fp32|fp16|int8] [--query learned|regex]

Three routes per prompt, all through the ONNX graphs only:

  frozen   lower + upper, nothing injected
  text     the calculator result spliced into the prompt as " [<result>]"
  latent   lower(prompt) -> query head -> calculator -> result symbols, then
           encoder(result, j) added to hidden16 at the position step j
           computes, before upper.  No tokens cross the boundary.

`--check` compares the latent route against demo.py's stream arm (the torch
path) on the same rows and prints exact agreement.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import regex_query as RQ

MAX_NEW = 16
N_OPERAND_SLOTS = 6
N_RESULT_SLOTS = 12
BLANK = 10
SIGN_OFFSET = 11
KIND_OFFSET = 13
OPS = ["add", "sub", "mul", "cmp"]
KINDS = ["numeric", "greater", "less", "equal"]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# ------------------------------------------------------------ pure-python bits
def decode_operand(slots):
    """Right-aligned digit slots -> int (port of model_utils.decode_operand)."""
    nz = [i for i, s in enumerate(slots) if s != BLANK]
    if not nz:
        return 0
    v = 0
    for i in range(max(nz), -1, -1):
        d = slots[i]
        v = v * 10 + (0 if d == BLANK else int(d))
    return v


def calculate(op_idx, a_slots, b_slots):
    op = OPS[int(op_idx)]
    a, b = decode_operand(list(a_slots)), decode_operand(list(b_slots))
    if op == "add":
        return str(a + b)
    if op == "sub":
        return str(a - b)
    if op == "mul":
        return str(a * b)
    return "greater" if a > b else ("less" if a < b else "equal")


def result_symbols_left(s):
    """Port of model_utils.result_symbols(..., align='left') for one string."""
    syms = [BLANK] * N_RESULT_SLOTS
    if s in ("greater", "less", "equal"):
        sign, kind = 0, KINDS.index(s)
    else:
        sign = 1 if s.startswith("-") else 0
        kind = 0
        digits = [int(c) for c in s if c.isdigit()]
        if len(digits) > N_RESULT_SLOTS:
            raise ValueError(s)
        for i, d in enumerate(digits):
            syms[i] = d
    return syms + [SIGN_OFFSET + sign, KIND_OFFSET + kind]


# ------------------------------------------------------------------ ORT driver
class Pipeline:
    def __init__(self, model_dir=MODEL_DIR, variant="fp32", threads=4,
                 providers=("CPUExecutionProvider",)):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = threads
        # ORT's default spin-wait burns the other cores between the many tiny
        # per-token runs this pipeline makes; turning it off is worth several
        # times the wall clock here.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        suf = "" if variant == "fp32" else "." + variant
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        def s(n):
            return ort.InferenceSession(os.path.join(model_dir, n), so,
                                        providers=list(providers))
        self.lower = s(f"lower{suf}.onnx")
        self.upper = s(f"upper{suf}.onnx")
        self.qh = s("query_head.onnx")
        self.enc = s("encoder.onnx")
        m = self.meta
        self.kv, self.hd = m["num_key_value_heads"], m["head_dim"]
        self.n_lo, self.n_up = m["n_lower_layers"], m["n_upper_layers"]
        self.eos = m["eos_token_id"]

    # -- caches
    def empty(self, n):
        z = np.zeros((n, 1, self.kv, 0, self.hd), np.float32)
        return z, z.copy()

    @staticmethod
    def mask(t, p):
        s = p + t
        q = np.arange(p, s)[:, None]
        return (np.arange(s)[None, :] <= q).astype(np.float32).reshape(1, 1, t, s)

    def run_lower(self, ids, pos, cache):
        pk, pv = cache
        h, nk, nv = self.lower.run(None, {
            "input_ids": np.asarray(ids, np.int64).reshape(1, -1),
            "position_ids": np.asarray(pos, np.int64).reshape(1, -1),
            "attn_mask": self.mask(len(ids), pk.shape[3]),
            "past_k": pk, "past_v": pv})
        return h, (nk, nv)

    def run_upper(self, hidden, pos, cache):
        pk, pv = cache
        lg, nk, nv = self.upper.run(None, {
            "hidden": hidden.astype(np.float32),
            "position_ids": np.asarray(pos, np.int64).reshape(1, -1),
            "attn_mask": self.mask(hidden.shape[1], pk.shape[3]),
            "past_k": pk, "past_v": pv})
        return lg, (nk, nv)

    def run_query_head(self, hidden, fp16_roundtrip=True):
        h = hidden.astype(np.float16).astype(np.float32) if fp16_roundtrip \
            else hidden.astype(np.float32)
        op, slot = self.qh.run(None, {
            "hidden": h,
            "mask": np.ones((1, hidden.shape[1]), np.int64)})
        return op[0], slot[0]

    def run_encoder(self, syms, step):
        return self.enc.run(None, {
            "symbols": np.asarray(syms, np.int64).reshape(1, -1),
            "step": np.asarray([step], np.int64)})[0]

    # -- generation
    def generate(self, ids, tok, syms=None, max_new=MAX_NEW):
        """Greedy decode.  `syms` non-None turns on the latent injection."""
        lo_c, up_c = self.empty(self.n_lo), self.empty(self.n_up)
        t = len(ids) - 1
        h, lo_c = self.run_lower(ids, list(range(len(ids))), lo_c)
        qh_hidden = h.copy()
        if syms is not None:
            h = h.copy()
            h[0, t, :] += self.run_encoder(syms, -1)[0]
        lg, up_c = self.run_upper(h, list(range(len(ids))), up_c)
        nxt = int(lg[0, -1].argmax())
        newline = tok("\n", add_special_tokens=False)["input_ids"][-1]
        gen = []
        for step in range(max_new):
            gen.append(nxt)
            if (nxt in (newline, self.eos)
                    and tok.decode(gen, skip_special_tokens=True).strip()):
                break
            if step == max_new - 1:
                break
            pos = [t + 1 + step]
            h, lo_c = self.run_lower([nxt], pos, lo_c)
            if syms is not None:
                h = h.copy()
                h[0, 0, :] += self.run_encoder(syms, step)[0]
            lg, up_c = self.run_upper(h, pos, up_c)
            nxt = int(lg[0, -1].argmax())
        txt = tok.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()
        return txt, gen, qh_hidden

    # -- the three routes
    def answer(self, prompt, tok, query="learned", tool_result=None):
        out = {"prompt": prompt}
        t0 = time.time()
        txt, gen, _ = self.generate(tok(prompt)["input_ids"], tok)
        out["frozen"] = {"text": txt, "tokens": len(gen),
                         "ms": (time.time() - t0) * 1000}

        # latent route: one prompt pass, reused for the query head
        t0 = time.time()
        ids = tok(prompt)["input_ids"]
        if query == "regex":
            q = RQ.extract(prompt)
            if q is None:
                out["latent"] = {"text": "", "tokens": 0, "ms": 0,
                                 "query": None, "calculator": None,
                                 "query_source": "regex"}
                return out
            op_name, a, b = q
            res = RQ.calculate(op_name, a, b)
            qtxt = f"{op_name} {a} {b}"
        else:
            lo_c = self.empty(self.n_lo)
            h, _ = self.run_lower(ids, list(range(len(ids))), lo_c)
            op, slot = self.run_query_head(h)
            oi = int(op.argmax(-1))
            slots = slot.argmax(-1).tolist()
            a = decode_operand(slots[:N_OPERAND_SLOTS])
            b = decode_operand(slots[N_OPERAND_SLOTS:])
            res = calculate(oi, slots[:N_OPERAND_SLOTS], slots[N_OPERAND_SLOTS:])
            qtxt = f"{OPS[oi]} {a} {b}"
        syms = result_symbols_left(res)
        txt, gen, _ = self.generate(ids, tok, syms=syms)
        out["latent"] = {"text": txt, "tokens": len(gen),
                         "ms": (time.time() - t0) * 1000, "query": qtxt,
                         "calculator": res, "query_source": query}

        t0 = time.time()
        r = tool_result if tool_result is not None else res
        tp = prompt + " [" + r + "]"
        txt, gen, _ = self.generate(tok(tp)["input_ids"], tok)
        out["text"] = {"text": txt, "tokens": len(gen),
                       "ms": (time.time() - t0) * 1000, "tool": " [" + r + "]"}
        return out


# ------------------------------------------------------------------ checking
def check_against_demo(pipe, tok, rows, query="learned"):
    """Compare the latent route with demo.py's stream arm (torch) row by row."""
    import demo as DEMO
    import eval as E
    import model_utils as mu
    import torch

    hf, tok = mu.load_model("smol")
    hook = mu.attach_hook(hf, pipe.meta["k"])
    enc = mu.ResultEncoder(hf.config.hidden_size, n_steps=mu.N_STREAM_STEPS)
    enc.load_state_dict(torch.load(os.path.join(
        mu.repo_dir(), "ckpt", "smol_stream-left_final.pt"))["state_dict"])
    enc.eval()
    qh_cache = E.load_query_head(hf, "smol", "attn")
    agree_text = agree_query = 0
    recs = []
    for r in rows:
        p = r["prompt"]
        op, slot = E.query_logits(hf, tok, [p], pipe.meta["k"], qh_cache[0],
                                  qh_cache[1], "attn")
        qtxt, calc = DEMO.decode_query(op[0], slot[0])
        syms = mu.result_symbols([calc], align="left")
        ref, _, _ = E.generate(hf, tok, [p], "stream", pipe.meta["k"], bs=1,
                               enc=enc, syms=syms, max_new=MAX_NEW)
        # latent route through ORT only: the other two routes cost time and
        # are not what this check compares
        ids = tok(p)["input_ids"]
        lo_c = pipe.empty(pipe.n_lo)
        h, _ = pipe.run_lower(ids, list(range(len(ids))), lo_c)
        op_l, slot_l = pipe.run_query_head(h)
        oi = int(op_l.argmax(-1))
        sl = slot_l.argmax(-1).tolist()
        got_calc = calculate(oi, sl[:N_OPERAND_SLOTS], sl[N_OPERAND_SLOTS:])
        got_q = (f"{OPS[oi]} {decode_operand(sl[:N_OPERAND_SLOTS])} "
                 f"{decode_operand(sl[N_OPERAND_SLOTS:])}")
        got_txt, _, _ = pipe.generate(ids, tok,
                                      syms=result_symbols_left(got_calc))
        agree_text += int(got_txt == ref[0])
        agree_query += int(got_q == qtxt and got_calc == calc)
        recs.append({"prompt": p, "gold": r.get("result_string"),
                     "demo": ref[0], "ort": got_txt, "demo_query": qtxt,
                     "ort_query": got_q})
        print(f"  {p!r:42} demo={ref[0]!r:>14} ort={got_txt!r:>14} "
              f"{'ok' if got_txt == ref[0] else 'MISMATCH'}", flush=True)
    hook.remove()
    return agree_text, agree_query, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--variant", default="fp32",
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--query", default="learned", choices=["learned", "regex"])
    ap.add_argument("--split", default="test_in")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--accuracy", type=int, default=0,
                    help="score N rows end to end (for the quant sweep)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    pipe = Pipeline(variant=args.variant)

    import random
    rows = D.load_split(args.split)
    rng = random.Random(args.seed)
    rows = rng.sample(rows, max(args.n, args.accuracy))

    out = {"variant": args.variant, "query": args.query}
    if args.accuracy:
        n_ok = 0
        t0 = time.time()
        for r in rows[:args.accuracy]:
            # latent route only -- this is the quantization gate, not a demo
            ids = tok(r["prompt"])["input_ids"]
            lo_c = pipe.empty(pipe.n_lo)
            h, _ = pipe.run_lower(ids, list(range(len(ids))), lo_c)
            op_l, slot_l = pipe.run_query_head(h)
            sl = slot_l.argmax(-1).tolist()
            res = calculate(int(op_l.argmax(-1)), sl[:N_OPERAND_SLOTS],
                            sl[N_OPERAND_SLOTS:])
            txt, _, _ = pipe.generate(ids, tok, syms=result_symbols_left(res))
            n_ok += int(txt == r["result_string"])
        out["accuracy"] = n_ok / args.accuracy
        out["n_accuracy"] = args.accuracy
        out["ms_per_answer"] = (time.time() - t0) / args.accuracy * 1000
        print(f"{args.variant} latent exact_match {out['accuracy']:.3f} "
              f"on {args.accuracy} rows "
              f"({out['ms_per_answer']:.0f} ms/answer, all three routes)")
    if args.check:
        a_txt, a_q, recs = check_against_demo(pipe, tok, rows[:args.n],
                                              args.query)
        out["agreement_text"] = f"{a_txt}/{args.n}"
        out["agreement_query"] = f"{a_q}/{args.n}"
        out["rows"] = recs
        print(f"latent route vs demo.py stream arm: text {a_txt}/{args.n}, "
              f"query {a_q}/{args.n}")
        for rec in recs:
            if rec["demo"] != rec["ort"]:
                print("  MISMATCH", rec)
    if not args.check and not args.accuracy:
        for r in rows[:args.n]:
            a = pipe.answer(r["prompt"], tok, query=args.query)
            print(f"{r['prompt']!r:44} gold={r['result_string']!r:>14} "
                  f"latent={a['latent']['text']!r:>14} "
                  f"frozen={a['frozen']['text']!r}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    print("DONE pipeline")


if __name__ == "__main__":
    main()
