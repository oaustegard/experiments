# Method

## Model and frame

PleIAs/Baguettotron, 321M parameters, 80 layers × 576 hidden, 65,536-token
vocabulary, `LlamaForCausalLM`, trained on the synthetic SYNTH corpus. CPU,
4 threads, fp32, greedy. Its chat template opens the assistant turn with a
`<think>` block.

The primary instrument is teacher-forced. Each item is a question whose answer is
one high-probability word, plus a fixed answer prefix that makes that word the
natural next token:

```
<|im_start|>user
Never mention the word Paris. What is the capital of France?<|im_end|>
<|im_start|>assistant
<think>
</think>
The capital of France is
```

Read the next-token distribution at that final position. One forward pass, no
sampling, so the only variance is between items. 0.29 s alone, 0.13 s batched.

That forced-empty think block is a real limitation, not a detail. The model card
says suppressing the think trace "significantly decreased performance for most
tasks", and if the machinery that acts on a prohibition runs inside the trace,
this frame measures a model that has not yet done the relevant work. Two
generation arms exist to check it: one that keeps the frame and lets the model
write the answer freely, and one that lets the model produce its own think block
from scratch.

## What is measured per cell

A scalar log P(forbidden word) cannot tell suppression apart from a prompt that
flattens the whole distribution, so each cell records four things: log P of the
target, its rank, the entropy of the full next-token distribution, and log P of a
control token — the highest-probability real-word alternative under no directive,
chosen per item.

Suppression is then defined in log-odds, with the control's shift subtracted:

```
suppression = [logodds(target|none) − logodds(target|cond)]
            − [logodds(control|none) − logodds(control|cond)]
```

A manipulation that lowers the target and the control equally scores zero. Raw
log P deltas are reported alongside, so the difference between the two framings
is visible rather than hidden.

## Separating case from token count

Capitalising a directive keyword costs a variable number of extra tokens in this
tokenizer, because the BPE merges were fitted on a corpus with almost no
capitalised words. `do not` → `DO NOT` costs nothing, both being two tokens.
`never` → `NE|VER` costs one. `avoid` → `AV|O|ID` costs two.
`under no circumstances` costs four.

So case and token count can be varied against each other inside one design. The
pilot used one keyword per cost level, which makes the cost a relabelling of
which word it is — the words differ in force, frequency and syntactic class, not
only in tokens. The main run uses 26 prohibitive keywords spread across five cost
bins, holding the sentence frame fixed, so the case effect is estimated within
each keyword and the slope across bins is estimated with several keywords per
bin.

Three further surface forms vary token count and case independently on the same
directive: markdown bold (`**never**`, more tokens, no case change), markdown
italic, alternating case (`nEvEr`, no extra tokens beyond fragmentation), and a
capitalised non-word of matched length, which tests whether an all-caps span acts
as an acronym cue rather than an emphasis cue.

## Baseline

The lowercase arm is **sentence case** — `Never mention the word Paris.` — not
all-lowercase. Directives begin sentences, and an all-lowercase directive is a
well-formedness violation rather than a neutral control; comparing CAPS against
it would measure typographic anomaly, not emphasis. An all-lowercase arm runs
separately on a subset, as the grammaticality control.

## Items

58 items, screened so the model puts the forbidden word's first token in the top
ten under no directive, and binned by baseline pressure — the model's intrinsic
log-probability of that word — into four strata: 12 low, 14 mid, 20 high, 12 at
ceiling. Rana's logistic pressure-violation relationship means an unstratified
pool reports a property of the pool, so the case effect is reported per stratum.

## Attention

Attention mass on a span is mechanically determined by the span's token count,
and attention weight is not causal importance. The correlational version of the
question has a trivially-true and a trivially-false answer depending on whether
total or per-token mass is reported. So the attention arm is a knockout: zero the
final query position's attention onto the directive span at one layer, re-run,
and re-read log P of the forbidden word. 80 layers, one forward pass each.
