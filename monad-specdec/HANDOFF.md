# Running this on an M5 Mac

Nothing large needs to move. The 3 GB of cached hidden states on the container
are deterministic given (model, corpus, order) and regenerating them will be
faster than downloading them. Clone the repo, fetch two parquet files, run.

```bash
git clone https://github.com/oaustegard/experiments
cd experiments/monad-specdec
pip install torch transformers pyarrow numpy

# Corpus. The SYNTH path is `default/partial-train/`, not `default/train/` —
# read it off the datasets-server /parquet endpoint rather than guessing.
curl -L -o synth0.parquet \
  "https://huggingface.co/datasets/PleIAs/SYNTH/resolve/refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet"
curl -L -o wiki0.parquet \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-raw-v1/train/0000.parquet"

python3 eagle_mac.py --verify          # do this first
python3 eagle_mac.py --tokens 20000000 --epochs 2
```

`eagle_mac.py` is self-contained and does not import the other scripts here.

## Run --verify first

The MPS path is untested. It was written on a Linux CPU container with no Apple
hardware, so `--verify` checks the three things that would silently corrupt
training if the device behaved differently: that the last hidden state is
actually what feeds the LM head, that embeddings are tied, and that a head
forward produces finite values on the device. On CPU all three pass and the
delta on the first is exactly 0.0.

If check 1 fails on MPS, stop. Everything downstream trains against the wrong
feature and the acceptance number will look plausible and be meaningless.

## Streaming instead of caching

The CPU pipeline cached hidden states to disk because a forward pass cost more
than re-reading 290 MB. That trade inverts when forward passes are cheap, and
the cache is what capped the CPU run: fp16 hidden states run 1,152 bytes per
token, so EAGLE's ~50M-token recipe would need **58 GB** against the 22 GB free
on the container.

`eagle_mac.py` streams instead — target forward and head update are fused, and
hidden states are never written. Disk cost is zero and token budget is bounded
only by time. `--cache` is not implemented; add it if you want many epochs over
a fixed corpus, which is the one case where storing wins.

## Data is no longer the constraint

Shard 0 of SYNTH alone holds roughly **126M English tokens**; all seven shards
are about **0.88B**. The CPU run consumed 2.75M, or **2.2% of one shard of seven**.

EAGLE's published recipe is on the order of 50M tokens. That is 18× what we
harvested and still only 40% of a single shard. Raise `--tokens` to add data; the corpus is already there.

## Where the numbers stand

Measured on 4 CPU cores, 4 epochs each, identical held-out set, so these are
comparable to each other:

| Train tokens | acceptance (α at γ=1) |
|---|---|
| 975k | 0.364 |
| 1.97M | 0.401 |
| 2.72M | (running) |

Roughly +0.037 per doubling. EAGLE reports 0.74–0.79 on 7B–70B targets. Naively
extrapolating to 50M tokens lands near 0.6. Eight doublings of log-linear
extrapolation is exactly the kind of projection that breaks, so treat that as a
reason to run it rather than as a forecast.

Two earlier numbers that do **not** compare to the table above: an α of 0.4095
from a 225k-token run and 0.207 end-to-end. Both used a different held-out set
(this repo's markdown rather than SYNTH), and a scaling curve is only valid
within one eval distribution.

## Measuring the speedup

`eagle_mac.py` prints sustained `tok/s` every 100 steps. The CPU baseline is
**431 tok/s** for the fused forward pass on 4 cores. That number is the one to
beat and the one that decides whether 50M tokens is an afternoon or a week.

Worth checking early, because it changes the plan: if the M5 sustains a few
thousand tok/s, the 50M-token run is a couple of hours and the interesting
question becomes γ>1 drafting quality rather than data volume.

## Then what

The end-to-end decoder is `eagle_e2e2.py`, which needs the repo's other files.
It measures acceptance and wall-clock against plain greedy on both
chat-templated prompts and raw prose, and asserts the output is token-identical
to greedy. Two known gaps in it:

- The draft head re-runs over its whole prefix each round instead of keeping a
  KV cache, so its wall-clock understates a real implementation. The projected
  column uses the measured cost ratio c = 0.049 and is fairer.
- The head is trained on next-token prediction only, with no feature-regression
  loss and no training-time feedback, so multi-step drafting degrades faster
  than a properly trained EAGLE would. Fixing that is the obvious next change.
