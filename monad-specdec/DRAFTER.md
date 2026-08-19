# Building an EAGLE drafter head for Baguettotron

I built one. It works, it is nowhere near good enough, and it did not need a
GPU. My earlier "needs a rented GPU" was the wrong shape of claim: the
constraint is training *data volume*, which on this container converts into
harvest wall-clock rather than into hardware.

## The head

Baguettotron, deciding what word comes next, ends up with 576 numbers per
position — a hidden state, the compressed form of everything it concluded from
the context. Those numbers are the last thing it computes before choosing. You
can check that they are the whole story: running the output projection on them
reproduces the model's logits to within 0.0.

An EAGLE head is a small module that reads that vector, plus the token the
target is about to emit, and guesses the token *after* it. Structurally:

    input   h_t          the target's hidden state at position t
            token_{t+1}  the token the target just chose
    module  FC(1152 -> 576), then one transformer decoder layer
    output  through the target's own frozen projection -> token_{t+2}

**4.2M trainable parameters**, 1.3% of the target's 321M. It borrows the
target's embedding and output projection and never updates them, and because
Baguettotron ties those two, it is borrowing one matrix rather than two.

It can be this small because it is not doing language modelling. The target
already did that; the head extrapolates one step from the target's own
conclusion. That is also why it needs the hidden state at all — take it away and
you are back to training a small language model from scratch, which is what
Monad effectively was.

## Training

The head's weights start as random numbers. Measured, before any training:
**acceptance 0.0000**, not one correct guess in 8,176 held-out positions.
Training is the loop that fixes that: show it a position, compare its guess to
the right answer, nudge every weight slightly in the direction that would have
helped, repeat.

The right answer is the interesting choice. Acceptance is defined as matching
the *target's* pick, not as being correct about English, so the labels are the
target's own argmax rather than the text. Both come out of the same forward
pass, so labelling costs nothing extra and no slow autoregressive generation is
needed.

## The run

| Stage | Measurement |
|---|---|
| Harvest 250k positions | 9.7 min at **431 tok/s**, 290 MB of fp16 hidden states |
| Train, 12 epochs | 11 min at **0.51 s/step**, 4 CPU cores |
| Acceptance, untrained | **0.0000** |
| Acceptance, trained, teacher-forced in-domain | **0.4095** |
| Acceptance, end-to-end free-running | **0.207** |
| EAGLE published, 7B-70B targets | 0.74–0.79 |

Every end-to-end run produced token-identical output to plain greedy decoding.

The 431 tok/s corrects the 192 tok/s in `RESOURCES.md`, which I measured while
other jobs were saturating the same four cores.

## Data or compute

Two arms on the same 225k-token harvest answer which one binds.

**More passes over fixed data** — acceptance climbs, then stops, while training
loss keeps falling:

| Epoch | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Acceptance | 0.305 | 0.344 | 0.373 | 0.418 | **0.427** | 0.423 |
| Train loss | 3.99 | 3.28 | 3.11 | 2.30 | 1.87 | 1.43 |

Past epoch 12 the head is memorizing the corpus. Loss falls by a quarter and
acceptance goes nowhere.

**More data at fixed passes** — still climbing, roughly +0.05 per doubling:

| Train tokens | 56k | 112k | 225k |
|---|---|---|---|
| Acceptance | 0.274 | 0.310 | 0.374 |

So compute saturates and data does not. EAGLE's published recipe is ~68k
ShareGPT dialogues, on the order of 50M tokens, about 200× what I harvested.
Extending the +0.05-per-doubling slope across those ~7.8 doublings lands near
0.75, which is where the published numbers sit. Eight doublings of log-linear
extrapolation is exactly the kind of projection that breaks, so treat the
agreement as encouraging rather than as evidence.

## So why a GPU

Not for the training. 4.2M parameters over 225k samples is 11 minutes on four
cores, and a GPU would be idle most of that time.

For the harvest, and only because of wall-clock. At the measured 431 tok/s,
50M tokens of hidden states takes **32 hours**, which outlives a CCotw session;
an A100 runs the same forward passes roughly two orders of magnitude faster.
Storage matters too — 50M positions of fp16 hidden states is ~58 GB, against the
26 GB free here.

The honest form of the claim: nothing about this needs specialised hardware. It
needs a machine that stays up for a day and has disk. Renting a GPU is simply
the cheapest way to buy that.

## Where it stands

The head does not beat baseline yet:

| γ | Acceptance | Measured speedup | Projected with a cached draft head |
|---|---|---|---|
| 1 | 0.207 | 0.96× | 1.15× |
| 2 | 0.116 | 0.90× | 1.03× |
| 3 | 0.081 | 0.71× | 0.95× |
| 4 | 0.060 | 0.68× | 0.89× |

Two caveats on that table, one against it and one for it.

Against: the measured column runs the head over its whole prefix every round
instead of keeping a KV cache for it, so it charges the draft far more than a
real implementation would. The projected column applies the measured acceptance
to the separately measured cost ratio c = 0.049 and is the fairer estimate.
Finding that bug was worth 53% of the acceptance rate on its own — feeding a
transformer layer one position with no history left its attention nothing to
attend to, and γ=1 acceptance read 0.135 until the head saw its own context.

For: free-running acceptance is 0.207 against 0.4095 teacher-forced in-domain,
and the split across prompts says why. The Python prompt scores 0.333 while the
prose prompts score 0.09–0.23, and the training corpus was this repository's
markdown and Python. Domain mismatch explains the gap, and more varied data
would close it.

## Files

| File | What |
|---|---|
| `eagle_harvest.py` | Harvest hidden states, tokens and target argmax → `/tmp/specdec/eagle_data.npz` |
| `eagle_train.py` | Train the 4.2M head, measure teacher-forced acceptance → `eagle_train.json` |
| `eagle_scaling.py` | Passes-versus-data arms → `eagle_scaling.json` |
| `eagle_e2e.py` | Greedy speculative decoding with the trained head → `eagle_e2e.json` |
