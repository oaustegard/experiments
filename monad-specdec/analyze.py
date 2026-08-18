"""Decompose the slowdown: token granularity, draft cost, and theory vs measured."""
import json, torch
from transformers import AutoTokenizer

torch.set_num_threads(4)
mt = AutoTokenizer.from_pretrained("PleIAs/Monad")
bt = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")

CORPUS = [
    "The Roman aqueducts were built to carry water into cities over long distances, "
    "relying on a steady gradient maintained across valleys and hills.",
    "Yeast consumes the sugars in dough and releases carbon dioxide, which is trapped "
    "by the gluten network and makes the loaf expand.",
    "In 1969 the Apollo 11 mission landed the first humans on the Moon, and Neil "
    "Armstrong stepped onto the surface while Michael Collins orbited above.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "Mitochondria generate most of the cell's supply of adenosine triphosphate, which "
    "is used as a source of chemical energy.",
]
chars = sum(len(t) for t in CORPUS)
m_toks = sum(len(mt(t, add_special_tokens=False).input_ids) for t in CORPUS)
b_toks = sum(len(bt(t, add_special_tokens=False).input_ids) for t in CORPUS)

lat = json.load(open("latency.json"))
d_ms = lat["runs"][0]["decode_ms_median"]
t_ms = lat["runs"][1]["decode_ms_median"]

gran = m_toks / b_toks           # draft steps needed to cover one target token
c_raw = d_ms / t_ms              # per-step cost ratio
c_eff = c_raw * gran             # cost of drafting one target token's worth


def theory(alpha, gamma, c):
    """Leviathan et al. expected speedup, with c as draft cost per target token."""
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


res = json.load(open("results.json"))
rows = []
for g, d in res["speculative"].items():
    a = d["mean_acceptance"]
    rows.append({
        "gamma": int(g), "measured_acceptance": a,
        "measured_speedup": d["mean_speedup"],
        "theory_speedup_raw_cost": round(theory(a, int(g), c_raw), 3),
        "theory_speedup_effective_cost": round(theory(a, int(g), c_eff), 3),
    })

# Break-even: the acceptance rate gamma=1 would need to reach parity.
def breakeven(c):
    return c  # (1+a)/(1+c) = 1  ->  a = c

out = {
    "tokenizer_granularity": {
        "corpus_chars": chars,
        "monad_tokens": m_toks, "baguettotron_tokens": b_toks,
        "monad_chars_per_token": round(chars / m_toks, 3),
        "baguettotron_chars_per_token": round(chars / b_toks, 3),
        "draft_steps_per_target_token": round(gran, 3),
    },
    "cost": {
        "draft_decode_ms": d_ms, "target_decode_ms": t_ms,
        "raw_cost_ratio_c": round(c_raw, 3),
        "effective_cost_ratio_c": round(c_eff, 3),
        "breakeven_acceptance_at_gamma1_raw": round(breakeven(c_raw), 3),
        "breakeven_acceptance_at_gamma1_effective": round(breakeven(c_eff), 3),
    },
    "table": rows,
}
print(json.dumps(out, indent=2))
json.dump(out, open("analysis.json", "w"), indent=2)
