# hypothetical-classification

**Question.** Doug Turnbull's "Don't classify. Hallucinate!"
([softwaredoug.com, 2026-08-10](https://softwaredoug.com/blog/2026/08/10/hypothetical-classifications))
proposes classifying into a closed vocabulary by asking a cheap model to invent a
plausible label and then snapping that invention onto the real label set with an
embedder. The model never sees the schema. Does it work, what does it cost, and where
does it stop working?

**Method.** WANDS (`wayfair/WANDS`) query → `product_class`: 860 legal labels, 468
queries carrying one gold label each. Second corpus: Muninn's memory store, 1,273 tags
used ≥3 times, 250 memories of 300-2000 characters, mean 4.8 gold tags. Generators:
`gemini-3.5-flash-lite` through the Cloudflare AI Gateway, and a Claude Code Haiku 4.5
subagent. Snappers: `all-MiniLM-L6-v2` and char-ngram (3-5) TF-IDF.

## Results

### WANDS, n=468, one gold label

| arm | acc@1 | acc@3 | input tok/query |
|---|---|---|---|
| TF-IDF: query → label | 0.316 | 0.453 | 0 |
| MiniLM: query → label | 0.417 | 0.564 | 0 |
| TF-IDF: written label, **novelty** prompt | 0.457 | 0.556 | 100 |
| MiniLM: written label, **novelty** prompt | 0.489 | 0.613 | 100 |
| TF-IDF: written label, **register** prompt | 0.528 | 0.620 | 100 |
| MiniLM: written label, **register** prompt | **0.564** | **0.690** | 100 |
| MiniLM: (query + written)/2, novelty | 0.500 | 0.639 | 100 |
| MiniLM: written label, novelty, batched ×40 | 0.496 | 0.641 | 6 |
| **structured output, all 860 labels shipped** | **0.701** | **0.744** | 5,265 |

Structured output emitted a verbatim-legal label on 99.1% of queries; snapping its output
anyway changed nothing (0.701 either way).

Wall clock at 3-way concurrency: unbatched written labels 507s, structured 512s,
batched ×40 **57s**.

### The prompt is the largest single variable

Same 40 WANDS queries, both generators, both snappers.

| prompt | generator | MiniLM @1 | @3 | TF-IDF @1 | @3 |
|---|---|---|---|---|---|
| — (embed the query) | — | 0.500 | 0.650 | 0.275 | 0.475 |
| novelty | gemini-3.5-flash-lite | 0.575 | 0.675 | 0.450 | 0.525 |
| novelty | Haiku 4.5 subagent | **0.100** | 0.275 | 0.150 | 0.225 |
| register | Haiku 4.5 subagent | 0.525 | **0.750** | 0.400 | 0.550 |

The post's prompt opens *"create a novel, never-seen-before classification"*. Haiku
obeyed it: `Hydraulic Styling Thrones`, `Weathered Branch-Frame Reflectors`,
`Chromatic Comfort Accents`, `Extinct Creature Shelf Companions`. Gemini flash-lite,
given the identical prompt, wrote `Salon & Styling Chairs`, `Rustic Wall Mirrors`,
`Turquoise Pillows` — it half-ignored the instruction, which is the only reason the
prompt works in the post.

Re-anchoring on register — *"write the label this vocabulary WOULD file the item under,
match the examples' register, do not worry whether it already exists"* — took Haiku from
0.100 to 0.525 and gave it the best acc@3 of any arm measured. It also beat the novelty
prompt on Gemini across all 468 (0.564 vs 0.489).

### Muninn tags, n=250, mean 4.8 gold tags

| arm | @1 | @3 | @5 |
|---|---|---|---|
| TF-IDF: summary → tag | 0.416 | 0.628 | 0.712 |
| TF-IDF: 5 written tags, novelty prompt | 0.208 | 0.352 | 0.424 |
| TF-IDF: 5 written tags, register prompt | 0.508 | 0.700 | 0.792 |
| TF-IDF: direct + register, interleaved | **0.672** | **0.852** | **0.888** |
| MiniLM: summary → tag | 0.356 | 0.584 | 0.656 |
| MiniLM: 5 written tags, novelty prompt | 0.244 | 0.380 | 0.460 |
| MiniLM: 5 written tags, register prompt | 0.464 | 0.676 | 0.792 |
| MiniLM: direct + register, interleaved | 0.640 | 0.836 | 0.888 |

Rows and generations are pinned in `muninn_tags_fixture.json`; the corpus is live and
grew by one memory mid-session, which moved an unpinned `random.sample` draw and every
number with it (see `ERRORS.md` #6).

Gemini's novelty-prompt tags: `remax-boundary`, `index-decoupling`, `s2orc-scale`,
`brute-force-quant`. Its register-prompt tags for the same memory: `correction`,
`architecture`, `vector-quantization`, `search`, `s2orc`.

### Can a 57M / 321M model be the hallucinating half? No.

Pleias `Monad` (57M) and `Baguettotron` (321M), both with `onnx-community` builds, so
the packaging premise holds: `Monad-q4f16` is **35 MB**, `Baguettotron-q4f16` is 236 MB,
`Xenova/all-MiniLM-L6-v2` int8 is 23 MB. A whole classifier fits in ~58 MB of browser
download. Neither model earns its bytes.

Monad has no chat template (a completion model, so few-shot `Query: / Category:` pairs);
Baguettotron has one that forces a `<think>` block. n=40, MiniLM snapper:

| generator | acc@1 | acc@3 | ms/query (CPU fp32) |
|---|---|---|---|
| none — snap the raw query | **0.500** | 0.650 | 0 |
| `gemini-3.5-flash-lite`, novelty prompt | 0.575 | 0.675 | ~1,100 |
| Monad 57M, few-shot | 0.425 | 0.500 | 645 |
| Baguettotron 321M, few-shot | 0.400 | 0.475 | 1,408 |
| Baguettotron 321M, chat template | — | — | 15,337 |

Both land **below** the no-model control. The outputs say why: `smart coffee table` →
`Smart coffee table`, `dinosaur` → `Dinosaur`, `turquoise pillows` → `Turquoise Pillows`.
They echo the query, which collapses the arm onto the direct-embed control, and then lose
ground to few-shot bleed — `chair and a half recliner` → `Chair & Recycling Bins`, copied
from the last exemplar. The chat arm never emitted `</think>` inside 220 tokens and
returned its own reasoning header 40 times out of 40.

Reranking instead of generating does not rescue it. Give the tiny model the embedder's
top-10 and score each candidate by `logP(label | few-shot, query)` — no format
compliance required, no way to emit an illegal label, which removes everything a 57M
model is worst at:

| | embedder alone | Monad rerank | Baguettotron rerank | ceiling (recall@10) |
|---|---|---|---|---|
| MiniLM top-10 | **0.500** | 0.325 | 0.350 | 0.825 |
| TF-IDF top-10 | 0.275 | 0.275 | 0.275 | 0.675 |

The headroom is real — the right label is in the embedder's top-10 for 82.5% of queries —
and neither model finds it more often than cosine argmax does.

### The in-browser classifier is the embedder alone

If the LM half subtracts, the encoder is the whole system. Full WANDS set, snapping the
raw query, no API call anywhere:

| encoder | int8 ONNX | acc@1 | acc@3 | acc@1 with a server-side register-prompt label |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 23 MB | 0.417 | 0.564 | 0.564 |
| bge-small-en-v1.5 | 33 MB | 0.427 | 0.583 | 0.541 |
| **gte-small** | **33 MB** | **0.455** | 0.594 | 0.571 |
| all-MiniLM-L12-v2 | 33 MB | 0.423 | 0.588 | 0.571 |
| bge-base-en-v1.5 | 109 MB | 0.462 | **0.630** | 0.575 |

`gte-small` is the size/accuracy knee: 0.455 acc@1 in **33 MB**, entirely client-side,
against MiniLM-L6's 0.417 for 10 MB less. `bge-base` buys +0.007 acc@1 for 3.3× the
download and is only worth it for acc@3 (0.630).

Those rows are fp32. Shipped as int8 ONNX through transformers.js — which is what the
33 MB figure actually buys — `gte-small` scores **0.434 / 0.588**, so quantisation costs
about two points at rank 1 and almost nothing at rank 3.

**Pin the encoder to `device: "wasm"`.** The same int8 weights on `device: "webgpu"`
return a collapsed space: every pair of labels ~0.995 cosine, ranking is noise, `Pillow`
answers *Wedding, Drains, Fabric, Flags, Candles* — silently, no error. fp32 PyTorch, the
same `model_quantized.onnx` under onnxruntime, and this page's own JavaScript on a CPU
backend all agree on *Standard Bed Pillows* at ~0.88, so the weights and the code are both
sound. `demo/index.html` therefore runs a 24-query smoke test on load and refuses to render
below 0.25 acc@3; it clears at 15/24.


## Findings

1. **The pattern beats every model-free baseline, and loses to shipping the vocabulary
   by 14 points.** 0.564 against 0.417 (direct MiniLM) and against structured output's
   0.701, at 6 input tokens per query versus 5,265. The post reports the first
   comparison and the cost, not the second comparison. The rule that follows: ship the
   vocabulary whenever it fits and the tokens are affordable; reach for this pattern
   when a provider enum cap, a 5,000-label taxonomy, or per-call cost at volume makes
   that impossible.

2. **Prompt for register, never for novelty — and the error hides behind a weak model.**
   Gemini flash-lite's partial disobedience masked a 7.5-point cost on WANDS and a
   30-point cost on the tag corpus. Haiku 4.5, following instructions properly, scored a
   fifth of the no-model control. A better instruction-follower is *worse* at the
   badly-worded prompt, which means swapping in a stronger cheap model silently breaks a
   deployment tuned on a weaker one. The instruction that is actually wanted is "do not
   worry whether the label exists", not "make it novel".

3. **The long-item "boundary" was the prompt.** Under the novelty prompt, the tag corpus
   read as a clean limit on the pattern — 0.208 against a 0.416 control, halved,
   unrecovered by asking for 5 labels instead of 1. That result was published in the
   first commit of both downstream artifacts and then withdrawn: the register prompt
   moves the same arm to 0.508, past the control. Long items amplify the register error
   (a distinctive vocabulary is where invented wording lands furthest from anything
   legal); they do not constitute a separate failure.

4. **Interleaving the direct snap with the written-label snap is worth +16.4pp on long
   items** (0.672 vs 0.508) and is redundant on short ones. The two rankings are
   complementary because a one-word label discards most of a 1,500-character document
   while the direct embedding keeps it.

5. **Batching 40 per call is free.** 0.496/0.641 against 0.489/0.613 unbatched, at 1/17
   the input tokens and 1/9 the wall clock. There is no accuracy argument for one item
   per call.

6. **A Claude Code subagent costs ~32,500 tokens before it reads its prompt.** A
   `general-purpose` Haiku 4.5 subagent asked to output the single word `ok`, zero tool
   calls, spent 32,539 tokens in 1,143 ms. Per item that floor is 813 tokens at batch 40
   and 32,500 at batch 1, which settles how any skill built on this pattern must call a
   subagent — in batches, or not at all.

7. **Char-ngram TF-IDF is a serious snapper, not a fallback.** 0.528 against MiniLM's
   0.564 on WANDS, and it beats MiniLM outright on the direct half of the tag corpus
   (0.416 vs 0.356) because a memory summary usually contains its own tag words
   literally. It needs no download and no GPU. Reach for MiniLM when items and labels
   share no surface wording.

## Shipped

- `oaustegard/muninn-utilities` [#127](https://github.com/oaustegard/muninn-utilities/pull/127)
  — `muninn_utils/hypothetical_classifier.py`, the pattern as a Python API with Gemini wired in.
- `oaustegard/claude-skills` [#782](https://github.com/oaustegard/claude-skills/pull/782)
  — `hallucinating-labels`, the same pattern for a Claude Code session, with `scripts/snap.py`.

## Files

| file | what |
|---|---|
| `bench.py` | WANDS loading, the novelty prompt, batched generation, snap + scoring |
| `run.py` | the five main WANDS arms → `results_lite.json`, `artifacts_lite.json` |
| `arm2.py` | structured-output baseline and batched hallucination |
| `register_prompt.py` | register vs novelty on Gemini, all 468 → `register_hall.json` |
| `score_haiku.py` | Haiku-subagent arms scored against Gemini on the same 40 |
| `muninn_tags.py` / `muninn_tags2.py` / `muninn_tags3.py` | first tag runs; superseded, rows not pinned |
| `muninn_tags_pinned.py` / `muninn_tags_fixture.json` | both tag arms with rows and generations pinned — the reported numbers |
| `haiku_arms.json` | the Haiku subagent's two label sets, verbatim |
| `tiny.py` / `tiny_arms.json` | Monad and Baguettotron as generators |
| `tiny_rerank.py` | the same two as likelihood rerankers over the embedder's top-k |
| `browser_embedders.py` | ONNX-available encoders, size against accuracy |
| `recheck.py` | re-derives every number in this file from the artifacts |

8. **A 57M-321M model cannot do the hallucinating half, at either interface.** As
   generators, Monad and Baguettotron score below the no-model control (0.425 and 0.400
   against 0.500); as likelihood rerankers over the embedder's top-10 they score 0.325
   and 0.350 against the same 0.500, with the gold label sitting in that top-10 for
   82.5% of queries. The mechanism follows from finding 2: **this pattern is a knowledge
   task wearing generation's clothes.** What `gemini-3.5-flash-lite` contributes is not
   reasoning but the fact that retailers write `Coffee Tables` and `Wall & Accent
   Mirrors` — a prior over how taxonomies name things, learned from web-scale
   pretraining. A SYNTH-trained reasoner has no retail taxonomy prior and reasoning
   capacity does not substitute for one. It is the case where small-reasoner-big-KB does
   not apply: the KB here is the label vocabulary, it is already attached, and the
   model's only job was the naming convention.

9. **A fully client-side classifier is real, and it is the encoder by itself.**
   `gte-small` int8 is 33 MB of ONNX and scores 0.455 acc@1 / 0.594 acc@3 on 860 labels
   with no API call. Adding Monad costs 35 MB, ~1.2 s/query, and 17 accuracy points.
   Adding a server-side register-prompt label instead buys +11.6pp (0.455 → 0.571), which
   is the honest price of leaving the browser.
