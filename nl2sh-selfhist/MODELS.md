# On-device generator bake-off — same rows, same eval, only the base differs

All three fine-tuned on the **identical 600 NL→command rows** and scored on the
**same independent cyber eval** (real commands, Gemini-authored NL that does not
name the utility), so the base model is the only variable. Greedy decoding, no
repetition penalty, unless noted.

| base | params | vocab | licence | train time | routing (leak-free, n=34) |
|---|---|---|---|---|---|
| Pleias-RAG-350M | 350M | 65k | Apache-2.0 | 25.5 min | 0.618 |
| Pleias-350M (no-RAG) | 268M | 65k | Apache-2.0 | 45.8 min | 0.529 |
| **Gemma 3 270M** | **268M** | **262k** | Gemma Terms | **15.9 min** | **0.706** |

**Gemma 3 270M is the best on-device generator measured — and by a real margin.**
+0.088 over Pleias-RAG at a *smaller* parameter count and the *fastest* training
of the three. That is the answer to "what about a Google model": yes, and it is
not a provenance-only choice — it is the most accurate base as well.

## Two things behind the number

**The big vocabulary is doing work.** Gemma's 262k vocab against Pleias' 65k
means more of a path or flag is a single token — the exact axis `monad-bsky`
found bounding small-model transcription. This is the arm that tests it, and the
routing lift is consistent with the vocabulary helping the model reproduce
argument tokens it would otherwise fragment.

**Raw greedy decoding degenerates on ~13% of outputs.** 5 of 38 Gemma outputs
repeat a token — `chown 0044 0044 0044 …`, `ssh 22 172.18.5.1 22 172.18.5.1 …`.
The leading utility still matches, so `utility_ok` counts them, which means the
0.706 *overstates* usable quality on those five rows. The obvious fix
overcorrected: `repetition_penalty=1.3` + `no_repeat_ngram_size=3` dropped
routing to **0.588** (`results_gemma_rp.json`), because forbidding any repeated
3-gram breaks legitimately-repetitive commands and pushes the model off the
correct leading token more often than it removes garbage. So neither penalised
nor raw is the deployment number; **the fair comparison is raw-vs-raw** — Gemma
0.706 against Pleias 0.618, both under identical greedy decoding — and finding
the decode sweet spot (a milder `repetition_penalty≈1.15`, no n-gram ban) is an
open tuning item, not a settled loss. The ~13% garbage rate is the real ceiling
to chip at, and it is a decoding problem, not a weights problem.

## Licence — the honest provenance note

Gemma ships under Google's **Gemma Terms of Use**, not Apache-2.0: permissive
for most use, but with a prohibited-use policy and terms to pass through, and the
`google/*` weights are gated (this arm used the ungated `unsloth` mirror).
Pleias is Apache-2.0, fewer strings. So "more acceptable than a Chinese model" is
a provenance-comfort judgement, not a licensing one — ironically Qwen2.5 is also
Apache-2.0. Gemma wins here on *accuracy*; the licence is a small point against,
not for.

## What this does not settle

Routing, not functional equivalence — `chown 0044 0044…` routes right and does
nothing. The n is 34. And the container had 60 man pages, so retrieval fed the
model a thin corpus; on a full machine the sources would be better and every
model's number would move. The relative ordering is the trustworthy part.

## Reproduce

```bash
python3 ../nl2sh-retrieval/gemma_arm.py train --tldr <tldr>/pages --nl2bash <nl2bash>/data/bash
python3 ../nl2sh-retrieval/gemma_arm.py eval  --tldr <tldr>/pages                      # raw
python3 ../nl2sh-retrieval/gemma_arm.py eval  --tldr <tldr>/pages --rep-penalty 1.3 --no-repeat 3 --tag _rp
```
