"""End-to-end greedy speculative decoding with the trained EAGLE head.

The head is a transformer layer trained over 511-token causal sequences, so it
must see its own history at inference. An earlier version of this script fed it
one position at a time with no cache, which left its attention with nothing to
attend to and cost two thirds of the acceptance rate. Here the head re-runs over
the full accumulated (feature, token) prefix each round, which matches training
exactly.

That re-run is O(prefix) per round where a real implementation would keep a KV
cache for the draft head, so the wall-clock numbers below understate what a
cached implementation would reach. The projected column uses the measured
acceptance with the separately measured cost ratio and is the fairer estimate.
"""
import json, time
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from eagle_train import EagleHead

torch.set_num_threads(4)
C_MEASURED = 0.049
cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                              dtype=torch.float32).eval()
embed, lm_head = target.model.embed_tokens, target.lm_head
head = EagleHead(cfg)
head.load_state_dict(torch.load("/tmp/specdec/eagle_head.pt"))
head.eval()

PROMPTS = [
    "The Roman aqueducts were built to carry water into cities. Their construction relied on",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The mission",
    "The mitochondrion is often described as the powerhouse of the cell because it",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
]
MAX_NEW = 48


def baseline(prompt, n=MAX_NEW):
    ids = tok(prompt, add_special_tokens=False).input_ids
    n0 = len(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True)
        past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        while len(ids) - n0 < n:
            ids.append(nxt)
            out = target(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
    return ids, time.perf_counter() - t0


def spec(prompt, gamma, n=MAX_NEW):
    ids = tok(prompt, add_special_tokens=False).input_ids
    n0 = len(ids)
    proposed = accepted = rounds = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        feats = out.hidden_states[-1][0]              # (p, H), feats[t] = h_t
        nxt = int(out.logits[0, -1].argmax())
        # Head input at t is (h_t, token_{t+1}); token_{t+1} for t<p-1 is ids[t+1].
        htoks = ids[1:] + [nxt]                       # aligned with feats

        while len(ids) - n0 < n:
            rounds += 1
            f_seq, t_seq, draft = feats, list(htoks), []
            for _ in range(gamma):
                a = head(f_seq.unsqueeze(0),
                         embed(torch.tensor([t_seq])))
                d = int(lm_head(a[:, -1:])[0, -1].argmax())
                draft.append(d)
                f_seq = torch.cat([f_seq, a[0, -1:]], dim=0)
                t_seq = t_seq + [d]
            proposed += len(draft)

            block = [nxt] + draft
            out = target(torch.tensor([block]), past_key_values=past,
                         use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            am = out.logits[0].argmax(-1)
            hs = out.hidden_states[-1][0]

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
    return {"ids": ids, "seconds": time.perf_counter() - t0, "rounds": rounds,
            "proposed": proposed, "accepted": accepted,
            "acceptance": accepted / proposed if proposed else 0.0}


def projected(a, g, c=C_MEASURED):
    return (1 - a ** (g + 1)) / ((1 - a) * (1 + g * c))


rows = []
for p in PROMPTS:
    b_ids, b_sec = baseline(p)
    n0 = len(tok(p, add_special_tokens=False).input_ids)
    for g in (1, 2, 3, 4):
        r = spec(p, g)
        rows.append({"prompt": p[:38], "gamma": g,
                     "speedup_uncached_draft": round(b_sec / r["seconds"], 3),
                     "acceptance": round(r["acceptance"], 3),
                     "projected_with_cached_draft": round(projected(r["acceptance"], g), 3),
                     "identical": r["ids"][:len(b_ids)] == b_ids[:len(r["ids"])]})
        print(rows[-1], flush=True)

by_g = {}
for g in (1, 2, 3, 4):
    sel = [r for r in rows if r["gamma"] == g]
    m = lambda k: round(sum(r[k] for r in sel) / len(sel), 3)
    by_g[str(g)] = {"mean_acceptance": m("acceptance"),
                    "mean_speedup_uncached_draft": m("speedup_uncached_draft"),
                    "mean_projected_cached_draft": m("projected_with_cached_draft"),
                    "all_identical": all(r["identical"] for r in sel)}
print(json.dumps(by_g, indent=2))
json.dump({"rows": rows, "by_gamma": by_g, "c_used": C_MEASURED},
          open("eagle_e2e.json", "w"), indent=2)
