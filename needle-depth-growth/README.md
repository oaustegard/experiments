# needle-depth-growth — can Cactus Needle 2 be grown to more layers after training?

Asked while reading Cactus's attention-only paper alongside `needle-bsky`: Monad
is 64 layers × 256 hidden and Needle 2 is 27 × 512, so could Needle be
"retro-trained" deeper the way a deep-narrow model is built?

**The surgery is trivial and is verified here. The training is the hard part,
and it is not a fine-tune.** `grow.py` grows the shipped 45M checkpoint to any
depth with identity-initialised blocks and shows the grown model's logits are
**byte-identical** to the original.

```
$ python3 grow.py 4
layers 27 -> 31   params 45,211,383 -> 49,615,207 (+4.40M, +9.7%)
logits max |delta| 0.000e+00   mean |logit| 197.317
argmax identical: True

$ python3 grow.py 21
layers 27 -> 48   params 45,211,383 -> 68,331,459 (+23.12M, +51.1%)
logits max |delta| 0.000e+00   mean |logit| 197.317
argmax identical: True
```

## Why it is this easy

The stack is an `nn.scan` (`Stack.__call__` in `needle/model/architecture.py`),
so **every per-layer tensor carries a leading axis of `num_layers`** — the block
weights inside the scanned collection and the MHC lane parameters as explicit
`(L, ...)` arrays. Adding depth is a concatenation along axis 0 plus a config
bump. Nothing downstream hardcodes 27: `decode.py` sizes the KV cache from
`cfg.num_layers` and `export.py` loops over it.

Two structural properties make appending safe:

* The MHC lane assignment is `arange(L) % 4`, so appending at the end leaves
  every existing layer's lane unchanged. Inserting in the *middle* would shift
  the lane of every later layer, and would need to be done in multiples of 4.
* `engram_layers=(2, 15)` is absolute, so the two Engram sites stay put.

## The identity recipe

A new layer is exactly the identity when four values are set. The scan body
computes `y = block(u) - u`, so a block that returns its input contributes
nothing:

| tensor | value | why |
|---|---|---|
| `block/attn_gate` | `-30.0` | `sigmoid` underflows to exactly 0 in fp16, killing the attention branch |
| `block/hadamard_mlp/d3` | `0.0` | the Hadamard branch returns exactly 0 |
| `mhc_a_res` | `0.0` | drops the input-dependent term from the lane-mixing logits |
| `mhc_b_res` | `40.0 * I` | `_sinkhorn` of that is the identity to ~1e-17, so `new_x = x` |

Everything else in the new slice is a copy of the last block, as depth
up-scaling normally does. `mhc_a_pre`/`a_post` and the phi projections are
irrelevant once `y` is 0 — `hpost` multiplies it.

Note the shipped `_res_identity_init` uses `4.0 * I`, which Sinkhorn maps to a
near-identity with ~1.8% off-diagonal mass. That is fine as an *initialisation*
and not fine as an identity, which is why this uses 40.

## What this does and does not answer

It answers the mechanical question: the checkpoint can be grown to arbitrary
depth, losslessly, in about 30 lines.

It does not make the model better. A grown model is exactly the old model until
the new layers are trained, and training them is pretraining, not fine-tuning:

* **The shipped trainer cannot do it.** `needle/model/finetune.py` is LoRA-only
  over a fixed `LORA_TARGETS` list. Low-rank adapters on identity-initialised
  blocks start from a zero-signal residual stream position; you would be writing
  a full-parameter JAX/Flax loop against `SimpleAttentionNetwork`.
* **The data is pretraining data.** `needle-bsky`'s fine-tune had 800 templated
  rows. Growing 27 → 31 adds 4.4M randomly-directed parameters; the paper's own
  size ladder moves loss with capacity only at a fixed 31.5B-token budget.
* **The paper argues against depth specifically.** Its component ablation puts
  depth at iso-param on a U-shape with a **20-layer optimum at this width**
  (8/20/32/48 layers → 2.168 / 2.1343 / 2.153 / 2.175 nats). Needle 2 ships at
  27, already past the optimum. That table is iso-*param*, so it is a
  depth-versus-width trade rather than a claim that more layers are bad — but it
  does say depth is not where this width wants its next parameters.
* **It costs the confidence head.** As with any change to the stack, the
  downstream `confidence_head` (8,192 parameters, never retrained) is reading a
  representation that moved. `needle-bsky` established that head as the reason
  to run a 45M model at all.
* **Untested here:** whether `export.py` / `quantize.py` will produce a working
  `.cact` at a non-standard depth, and whether the CQ2 weight-bit spec
  (`embedding=4,mhc=4,default=2`) survives it.

And the premise is worth separating: Monad is 64 × 256 **by pretraining design**,
not by retro-fitting layers onto a smaller model, so it is not a precedent for
growth. `monad-bsky` measured that shape at this task and it lost — 0.481
routable fine-tuned against Needle's 0.611 base.

## Running it

```bash
python3 -m pip install --break-system-packages cactus-needle
python3 grow.py 4          # 27 -> 31 layers
python3 grow.py 21         # 27 -> 48, the deepest stack the paper reports training
```

Weights are fetched from `Cactus-Compute/needle2` on first use. CPU only,
~11 s per check.
