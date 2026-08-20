# Which architecture — measured, not argued

The question: *"we should have SOME model; should we just do lexical retrieval
with a Gemini flash-lite generator?"* Answered on the independent eval (real
cyber commands, Gemini-authored NL that does not name the utility, leak-free
n=35).

## The measurement that decides it

The retrieval tier exists to feed documentation to a model too small to know the
commands. So the prior question is: does a cloud generator need it at all? Give
Gemini the request and **nothing else** — no tldr, no man, no distractors — and
score utility routing against the retrieval-fed small model.

| approach | routing (leak-free) | docs fed? | command rate |
|---|---|---|---|
| fine-tuned Pleias-350M + BM25 retrieval | 0.618 | yes (k=3) | 0.84 |
| Gemini 3.7-flash, direct | 0.743 | **no** | 1.00 |
| **Gemini 3.5-flash-lite, direct** | **0.771** | **no** | 1.00 |

**The cloud model with no retrieval beats the retrieval-fed small model.** And
the cloud numbers are *depressed* by the corpus: Gemini safety-refuses several
offensive-security commands (*"Sorry, I cannot fulfill your request to crack
passwords"*), each scored as a miss. On a non-security distribution both would
be higher.

So for a cloud generator, **lexical retrieval adds nothing to accuracy.** A
capable model already knows `find`, `grep`, `chmod`, `tar`. Retrieval's only
remaining jobs are the long tail the model does not know and flag-grounding —
and our retrieval is weakest exactly on the tail (0.233 recall@5).

## What that means for the build

The whole retrieval-plus-small-model stack was solving one problem: *make a tiny
model viable offline.* If cloud is acceptable, that problem does not exist.

**Cloud path — the model is primary, retrieval is a fallback.**
Prompt Gemini directly. Do not gate it behind lexical retrieval. Add retrieval
only as a *conditional* layer: when the model abstains, or when the leading
utility is rare enough that grounding is worth a round trip. This is the cascade
shape inverted — the capable layer runs first, the cheap layer catches its gaps.

**Offline / private / zero-cost path — the small model, and retrieval earns its
keep.** The fine-tuned 350M at 0.618 with retrieval is the only option with no
network, and there the docs matter because the model is genuinely weak. ~12 s
per query on 4 CPU cores unquantised; far faster on Metal/GPU.

## On flash-lite specifically — it is the clean winner

flash-lite is the most accurate here (0.771) AND the fastest AND the cheapest
(the high-volume model), with **no thinking tax**. An earlier version of this
memo claimed it carried a mandatory thinking cost; that was wrong. The confusion:
`gemini-3.5-flash-lite` 400s on an explicit `thinkingBudget: 0`, which reads like
"thinking is forced on" — but **omitting `thinkingConfig` entirely gives
`thoughtsTokenCount = 0`** and a ~1.0-1.3 s response (that latency is the gateway
round-trip, identical for flash and flash-lite). Passing `thinkingBudget: -1`
turns dynamic thinking ON (213 tokens), which is what you must NOT do for
high-volume work.

The 0.771 above was already measured with thinking off: `gemini_direct.py --think -1`
routes through `generate(thinking_budget=-1)`, which omits `thinkingConfig` (the
fast path), so flash-lite ran at 0 thinking tokens. So the recommendation is
simply: **use `gemini-3.5-flash-lite` with no `thinkingConfig`** — most accurate,
no thinking, high-volume-priced. `gemini-3.7-flash` (0.743, `thinkingBudget: 0`)
is the fallback if you specifically want that model.

## Caveats

- **n=35 leak-free**, one security-heavy domain, and the safety refusals hit the
  cloud models specifically. Re-run on a general-purpose command distribution
  before committing.
- **Utility routing, not functional equivalence.** These models pick the right
  utility 62–77% of the time; whether the flags are right is a further,
  strictly-lower number (`funceq.py`, blocked on most cyber commands needing a
  network/security sandbox).
- **The retrieval-fed number uses the fine-tuned small model, not Gemini.** A
  Gemini-plus-retrieval arm was not run because the direct number already
  exceeds the small-model-plus-retrieval stack, which answers the question; if
  retrieval *helped* Gemini it would only widen the cloud path's lead.

## Reproduce

```bash
python3 gemini_direct.py --model gemini-3.7-flash
python3 gemini_direct.py --model gemini-3.5-flash-lite --think -1   # -1 omits thinkingConfig -> thinking OFF (fast path)
python3 run_independent_eval.py --model ../nl2sh-retrieval/ft       # the small-model-plus-retrieval baseline
```
