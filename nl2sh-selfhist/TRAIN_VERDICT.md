# Is it worth training Gemma? Yes — because of the abstention gate.

Both models scored on the independent cyber eval, same fence-aware parser, under
three source conditions:

| condition | fine-tuned Gemma | untrained Gemma-3-270m-it |
|---|---|---|
| oracle (gold always in sources) | **0.706** | 0.500 |
| none (no sources) | 0.000 | **0.176** |
| real BM25 retrieval | **0.206** | 0.147 |

Three facts, and they resolve the question:

1. **Training helps a lot when sources are good: +0.206** (0.500 → 0.706). The
   fine-tuned model copies the right command out of retrieved docs far better
   than the instruction-tuned base does.
2. **Training destroys the fallback: 0.000 without sources**, where the untrained
   model retains general knowledge (0.176 — it can do `cd`, `ls`, `cat` from
   nothing). Fine-tuning turned a generalist into a pure copy-from-context
   machine.
3. **On the raw real system the fine-tuned model still wins** (0.206 vs 0.147):
   its much better use of the 26% of queries where retrieval hits outweighs its
   total failure on the 74% where it misses.

**The abstention gate makes the decision lopsided.** nlsh only generates when
retrieval is *confident* (margin ≥ 5), i.e. only on queries where good sources
are present — and abstains (shows pages) otherwise. On that confident subset the
fine-tuned model scores ~0.706 against the untrained ~0.500, and its
0.000-without-sources brittleness **never fires**, because we never run it
sourceless — we abstain instead.

So under the architecture we actually chose:

> **Train.** The fine-tuned model is strictly better on the only path where the
> model runs (confident retrieval → generate), and the downside of training —
> collapse without sources — is exactly what the abstention gate already
> prevents.

The graceful degradation of the untrained model (0.176 from nothing) would only
matter in a *no-abstention* design that runs the model on every query. That is
the design this repo spent the night arguing against.

## The larger point this does not change

Training buys +0.206 **on the confident subset**, but the confident subset is
small because retrieval is weak (gold in sources 26% of the time). The whole-
system number is 0.206, and it is retrieval-bound, not model-bound. Training is
worth it, and it is second-order: **the first-order lever is still retrieval** —
dense to fix semantic misses, page-level granularity now that Gemma's 32k
context removes the chunking constraint, and a query-reformulation step before
retrieval. Train the model, but spend the next effort on the sources it reads.
