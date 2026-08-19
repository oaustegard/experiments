"""End-to-end speculative decoding, evaluated in and out of distribution.

Baguettotron is a reasoning model: its chat template opens the assistant turn
with <think>, and its own generations follow SYNTH's reasoning format. The first
end-to-end run prompted it with bare prose continuations, which is not how it is
used and not what the harvest looks like. This runs both prompt styles so the
gap is visible rather than assumed.

Usage: eagle_e2e2.py <head_checkpoint>
"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from eagle_train import EagleHead

torch.set_num_threads(4)
C_MEASURED = 0.049
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/specdec/eagle_head.pt"
cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                              dtype=torch.float32).eval()
embed, lm_head = target.model.embed_tokens, target.lm_head
head = EagleHead(cfg)
head.load_state_dict(torch.load(CKPT))
head.eval()

QUERIES = [
    "Why does bread rise when yeast is added?",
    "How did Roman engineers keep aqueducts flowing across valleys?",
    "What does a mitochondrion actually do in a cell?",
    "Write a Python function that returns the nth Fibonacci number.",
]
RAW = [
    "The Roman aqueducts were built to carry water into cities. Their construction relied on",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The mission",
    "The mitochondrion is often described as the powerhouse of the cell because it",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
]
MAX_NEW = 48


def baseline(ids, n=MAX_NEW):
    ids = list(ids); n0 = len(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True)
        past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        while len(ids) - n0 < n:
            ids.append(nxt)
            out = target(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
    return ids, time.perf_counter() - t0


def spec(ids0, gamma, n=MAX_NEW):
    ids = list(ids0); n0 = len(ids)
    proposed = accepted = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        feats = out.hidden_states[-1][0]
        nxt = int(out.logits[0, -1].argmax())
        htoks = ids[1:] + [nxt]
        while len(ids) - n0 < n:
            f_seq, t_seq, draft = feats, list(htoks), []
            for _ in range(gamma):
                a = head(f_seq.unsqueeze(0), embed(torch.tensor([t_seq])))
                d = int(lm_head(a[:, -1:])[0, -1].argmax())
                draft.append(d)
                f_seq = torch.cat([f_seq, a[0, -1:]], dim=0)
                t_seq = t_seq + [d]
            proposed += len(draft)
            block = [nxt] + draft
            out = target(torch.tensor([block]), past_key_values=past,
                         use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            am = out.logits[0].argmax(-1); hs = out.hidden_states[-1][0]
            ids.append(nxt)
            k = 0
            while k < len(draft) and int(am[k]) == draft[k] and len(ids) - n0 < n:
                ids.append(draft[k]); k += 1
            accepted += k
            if len(ids) - n0 >= n:
                break
            new_nxt = int(am[k])
            feats = torch.cat([feats, hs[:k + 1]], dim=0)
            htoks = htoks + block[1:k + 1] + [new_nxt]
            nxt = new_nxt
            past.crop(len(ids))
    return {"ids": ids, "seconds": time.perf_counter() - t0,
            "acceptance": accepted / proposed if proposed else 0.0}


def projected(a, g, c=C_MEASURED):
    return (1 - a ** (g + 1)) / ((1 - a) * (1 + g * c))


def run(name, prompt_ids):
    rows = []
    for pid in prompt_ids:
        b_ids, b_sec = baseline(pid)
        for g in (1, 2, 3, 4):
            r = spec(pid, g)
            rows.append({"style": name, "gamma": g,
                         "acceptance": round(r["acceptance"], 3),
                         "speedup_uncached_draft": round(b_sec / r["seconds"], 3),
                         "projected_cached_draft": round(projected(r["acceptance"], g), 3),
                         "identical": r["ids"][:len(b_ids)] == b_ids[:len(r["ids"])]})
    agg = {}
    for g in (1, 2, 3, 4):
        sel = [x for x in rows if x["gamma"] == g]
        m = lambda k: round(sum(x[k] for x in sel) / len(sel), 3)
        agg[str(g)] = {"acceptance": m("acceptance"),
                       "speedup_uncached_draft": m("speedup_uncached_draft"),
                       "projected_cached_draft": m("projected_cached_draft"),
                       "all_identical": all(x["identical"] for x in sel)}
    print(name, json.dumps(agg, indent=2), flush=True)
    return rows, agg


chat_ids = [tok(tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids for q in QUERIES]
raw_ids = [tok(p, add_special_tokens=False).input_ids for p in RAW]

r1, a1 = run("chat_in_distribution", chat_ids)
r2, a2 = run("raw_prose_out_of_distribution", raw_ids)
json.dump({"checkpoint": CKPT, "c_used": C_MEASURED,
           "chat_in_distribution": a1, "raw_prose_out_of_distribution": a2,
           "rows": r1 + r2}, open("eagle_e2e2.json", "w"), indent=2)
