"""Shared harness for the capitalised-emphasis probes on Baguettotron.

Primary instrument is teacher-forced: build a chat prompt, force the assistant
turn to a fixed answer frame, and read the log-probability the model assigns to
the forbidden word at the position where it is the natural continuation. One
forward pass, no sampling, no decoding variance.
"""
import os, json, math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "PleIAs/Baguettotron"
THREADS = int(os.environ.get("CAPS_THREADS", "4"))

_state = {}


def load(eager=False):
    """eager=True forces the attention implementation that actually returns
    attention weights; sdpa silently returns None for them."""
    key = "m_eager" if eager else "m"
    if key in _state:
        return _state["tok"], _state[key]
    torch.set_num_threads(THREADS)
    if "tok" not in _state:
        _state["tok"] = AutoTokenizer.from_pretrained(
            MODEL, clean_up_tokenization_spaces=False)
    kw = dict(dtype=torch.float32)
    if eager:
        kw["attn_implementation"] = "eager"
    _state[key] = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    return _state["tok"], _state[key]


def ntok(s):
    tok, _ = load()
    return len(tok.encode(s, add_special_tokens=False))


def build_prompt(user_content, answer_prefix):
    """Chat prompt with the assistant turn forced past the think block."""
    tok, _ = load()
    head = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True)
    return head + "</think>\n" + answer_prefix


@torch.no_grad()
def logprobs(prompts, targets, batch_size=8):
    """Log-prob of each target's FIRST token at the final position of its prompt.

    Left-padded batching; padding is masked out of attention so a padded row
    scores identically to the same row run alone (asserted in test_caps_lib.py).
    """
    tok, m = load()
    pad = tok.encode("[PAD]", add_special_tokens=False)[0]
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        tgt = targets[i:i + batch_size]
        enc = [tok.encode(p, add_special_tokens=False) for p in chunk]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, L - len(e):] = torch.tensor(e)
            att[j, L - len(e):] = 1
        lg = m(input_ids=ids, attention_mask=att).logits[:, -1, :]
        lp = torch.log_softmax(lg.float(), -1)
        for j, t in enumerate(tgt):
            tid = tok.encode(t, add_special_tokens=False)[0]
            out.append(lp[j, tid].item())
    return out


@torch.no_grad()
def generate(user_content, max_new_tokens=400, seed=None):
    """Greedy continuation of the full assistant turn, think block included."""
    tok, m = load()
    head = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(head, return_tensors="pt", add_special_tokens=False).input_ids
    if seed is not None:
        torch.manual_seed(seed)
        out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.8, top_p=0.95, pad_token_id=pad_id(tok))
    else:
        out = m.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=pad_id(tok))
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)


def pad_id(tok):
    return tok.encode("[PAD]", add_special_tokens=False)[0]


def span_range(prompt, span):
    """Token index range of `span` inside `prompt`, or None if it does not
    tokenise as a contiguous block there.  Returns an out-of-band None rather
    than a (0, 0) that a caller would happily index with."""
    tok, _ = load()
    idx = prompt.find(span)
    if idx < 0:
        return None
    before = len(tok.encode(prompt[:idx], add_special_tokens=False))
    inside = len(tok.encode(prompt[:idx + len(span)], add_special_tokens=False))
    if inside <= before:
        return None
    return before, inside


@torch.no_grad()
def attention_to_span(prompt, span):
    """Mean attention mass from the final query position onto `span`, per layer.

    Returns (per_layer_total, per_layer_per_token, span_len) or None if the span
    does not resolve.
    """
    tok, m = load(eager=True)
    rng = span_range(prompt, span)
    if rng is None:
        return None
    a, b = rng
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    out = m(ids, output_attentions=True)
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError("model returned no attention weights; need eager attn")
    totals = []
    for layer in out.attentions:            # (1, heads, q, k)
        mass = layer[0, :, -1, a:b].sum(-1).mean().item()
        totals.append(mass)
    n = b - a
    return totals, [t / n for t in totals], n


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(a, b):
    """Paired t on differences a-b. Returns (mean_diff, t, n)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return mean(d), float("nan"), n
    s = stdev(d)
    if s == 0:
        return mean(d), float("inf") if mean(d) else 0.0, n
    return mean(d), mean(d) / (s / math.sqrt(n)), n


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", path)
