"""Cross-tokenizer speculative decoding: Monad (56M, 8K vocab) drafts for
Baguettotron (321M, 65K vocab).

The two models share an architecture family and a training corpus but not a
tokenizer, so draft tokens cannot be verified directly against target logits.
This uses string-level exact match (Timor et al. 2025): the draft's token
sequence is detokenized to text, re-tokenized under the target vocabulary, and
verified token-by-token against the target's own greedy argmax. Output is
identical to plain greedy decoding with the target, so any wall-clock
difference is pure overhead or pure win.

Both models keep a KV cache over their own token stream; on each commit the
cache is truncated to the longest shared token prefix and only the divergent
suffix is re-run.
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT_REPO = "PleIAs/Monad"
TARGET_REPO = "PleIAs/Baguettotron"


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class CachedRunner:
    """Wraps a causal LM with a KV cache keyed to a token-id list.

    Re-tokenizing a growing string can change token boundaries near the end, so
    the cache is addressed by token ids rather than by length: on each call the
    cache is cropped to the longest prefix it shares with the requested ids.
    """

    def __init__(self, repo, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
        self.model.eval()
        self.cache = None
        self.ids = []
        self.forward_calls = 0
        self.forward_tokens = 0

    def logits_for(self, ids, need_from=None):
        """Run `ids`, reusing the cached prefix.

        Returns (logits, offset): logits[j] predicts the token after absolute
        position offset + j. `need_from` forces the run to start no later than
        that absolute position, so the caller gets logits it needs even when
        the cache already covers them.
        """
        keep = _common_prefix(self.ids, ids)
        if keep == len(ids):
            keep = len(ids) - 1  # always run at least the final token
        if need_from is not None:
            keep = min(keep, need_from)
        if self.cache is not None and keep < len(self.ids):
            self.cache.crop(keep)
        elif self.cache is None:
            keep = 0
        new = ids[keep:]
        inp = torch.tensor([new])
        with torch.no_grad():
            out = self.model(inp, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.ids = list(ids)
        self.forward_calls += 1
        self.forward_tokens += len(new)
        return out.logits[0], keep

    def greedy_next(self, ids):
        logits, _ = self.logits_for(ids)
        return int(logits[-1].argmax())


def draft_tokens(drafter, text, gamma):
    """Greedily extend `text` by `gamma` draft tokens; return the new text."""
    ids = drafter.tok(text, add_special_tokens=False).input_ids
    for _ in range(gamma):
        ids.append(drafter.greedy_next(ids))
    return drafter.tok.decode(ids)


def speculative_generate(drafter, target, prompt, max_new_tokens=48, gamma=4):
    tt = target.tok
    prompt_ids = tt(prompt, add_special_tokens=False).input_ids
    ids = list(prompt_ids)
    proposed = accepted = rounds = 0
    t0 = time.perf_counter()

    while len(ids) - len(prompt_ids) < max_new_tokens:
        rounds += 1
        text = tt.decode(ids)
        cand_text = draft_tokens(drafter, text, gamma)
        cand_ids = tt(cand_text, add_special_tokens=False).input_ids

        # Keep only candidate tokens that extend the committed prefix.
        shared = _common_prefix(ids, cand_ids)
        if shared < len(ids):
            # Draft's text re-tokenized differently inside committed region;
            # fall back to a plain target step this round.
            cand_ids = ids
        extra = cand_ids[len(ids):]
        proposed += len(extra)

        logits, offset = target.logits_for(ids + extra, need_from=len(ids) - 1)
        # logits[j] predicts the token after absolute position offset + j.
        base = len(ids) - 1 - offset
        n_ok = 0
        for i, tid in enumerate(extra):
            if int(logits[base + i].argmax()) == tid:
                n_ok += 1
            else:
                break
        accepted += n_ok
        bonus = int(logits[base + n_ok].argmax())
        ids = ids + extra[:n_ok] + [bonus]

    # A round commits accepted + bonus, so the loop can overshoot the budget.
    ids = ids[:len(prompt_ids) + max_new_tokens]
    elapsed = time.perf_counter() - t0
    return {
        "text": tt.decode(ids),
        "ids": ids,
        "new_tokens": len(ids) - len(prompt_ids),
        "seconds": elapsed,
        "rounds": rounds,
        "proposed": proposed,
        "accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "target_forward_tokens": target.forward_tokens,
        "draft_forward_tokens": drafter.forward_tokens,
    }


def baseline_generate(target, prompt, max_new_tokens=48):
    tt = target.tok
    ids = tt(prompt, add_special_tokens=False).input_ids
    n0 = len(ids)
    t0 = time.perf_counter()
    while len(ids) - n0 < max_new_tokens:
        ids.append(target.greedy_next(ids))
    return {"text": tt.decode(ids), "ids": ids, "new_tokens": len(ids) - n0,
            "seconds": time.perf_counter() - t0}
