# Errors — monad-bsky

What was wrong, how it surfaced, which way it pushed the conclusion.

## 1. The stated cause of the copying failure was wrong

**What.** The draft explained tuned Monad's 0.512 verbatim-copy accuracy as a
tokenizer problem: "an 8,192-piece vocabulary learned from prose shatters
`austegard.com` into long subword sequences." It read well, it fitted the
symptom, and it was in the file before I checked it.

**Cause.** I never measured Needle's vocabulary. I assumed a purpose-built
tool-caller would carry a larger one and wrote "~32k (SentencePiece)" into the
comparison table from nothing.

**How caught.** Verifying that table entry before publishing.
`SANTokenizer.vocab_size` is **8,192** — identical to Monad's. Segmenting ten
identifiers through both tokenizers gives 111 Needle pieces against 109 Monad
pieces, and `austegard.com` is `['a','ust','eg','ard','.','com']` in each.

**Fix.** The measurement replaced the explanation. The cause that survives is
the training objective: Needle's contract is that arguments carry only values
evidenced by the input, and its **base** weights — which never saw this
experiment's data — already copy at 0.780, where three epochs of fine-tuning
left Monad at 0.512.

**Direction.** Would have shipped a confident mechanistic claim, in a writeup
whose whole point is that mechanism, resting on a number I invented. The
corrected version is a stronger result: the two models are tokenization-matched,
so the gap is attributable to what they were trained to do.

## 2. Needle's layer count, also invented

**What.** "Needle is 22 layers wide by comparison" appeared in the latency
section as an explanation for Monad's 64-layer slowness.

**Cause.** Same reflex as #1 — reaching for a contrast the sentence wanted
rather than a number I had. Needle's engine does not expose a layer count
through the Python surface and I never looked at the paper.

**Fix.** Replaced with what is actually known: Needle runs a hand-written C++
engine on quantized weights while this is `transformers` on fp32 tensors, so the
latency comparison is implementation as much as architecture.

**Direction.** Caught in the same pass as #1, before publishing. Two invented
numbers in one draft is the finding worth keeping about this session.

## 3. `nohup ... &` inside a background tool call

**What.** The first training launch used `nohup python3 train.py ... &` inside a
`run_in_background` Bash call, which returns as soon as the shell exits and
leaves the trainer orphaned.

**How caught.** Immediately, before it mattered — this is the failure documented
as `needle-bsky/ERRORS.md` #3 and #5, from the previous session, and it fired
again the very next day in a slightly different costume.

**Fix.** The launcher runs the trainer in the foreground *of* a backgrounded
tool call. `pgrep -f | xargs -r kill` for the cleanup, never `pkill -f`.

**Direction.** Cost one restart. The entry exists because knowing the rule did
not prevent the reflex.

## 4. Running the epoch-1 eval alongside training

**What.** Started the epoch-1 evaluation while epochs 2 and 3 were still
training, on 4 cores. Training's per-step time went from ~9s to ~19s and the
eval had produced nothing after 12 minutes; I killed the eval and ran everything
in sequence afterwards.

**Cause.** Wanting an early read. The parent experiment already measured what
contention does to this box (`needle-bsky/ERRORS.md` #6).

**Direction.** Cost roughly 15 minutes of wall-clock, no measurement. All
reported latencies come from the sequential run on an otherwise idle box.

## 5. Validation loss was never going to answer the question

**What.** The training script reports validation loss per epoch, and it looks
excellent: 0.0128 → 0.0040 → 0.0017. Routable accuracy over the same three
checkpoints goes 0.389 → 0.481 → 0.444.

**Cause.** The 80-row validation split is drawn from the same 800 templated rows
as the training set, so it measures template fit, not task generalisation. This
is not a bug in the code — it is a bug in reading the number as if it were
progress.

**Fix.** Nothing in the code. It is called out in RESULTS.md because the split
looks like a held-out set and is not one, and because the eval set — a
differently-phrased 62 queries — is the only number that moved in a way worth
believing.

**Direction.** Would have supported "the third epoch is the best model" on the
strength of the loss curve. The eval says epoch 2 is, by a margin that does not
reach significance either way.
