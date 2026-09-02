# Tier 2: a better inverter, trained on the M5 Max, shipped to the browser

The 40k-pair proof of concept recovered the exact string 2.4% of the time from
the float vector (`RESULTS.md`). This is the plan to raise that enough for a live
in-browser demo. It fixes what held the PoC back, the training budget, on Apple
Silicon, then exports to ONNX for `austegard.com`.

The code changes this plan needs are already on this branch: `train.py` takes
`--device mps --bf16`, resumes mid-epoch, and reports greedy verifier cosine per
epoch; `build_data.py` takes larger sentence corpora; `export_onnx.py` produces
verified browser graphs. Every step below is therefore a command with real arguments.

## Why the PoC was weak, in one number

The zero-step base scored verifier cosine **0.549 on its own training set** and
0.547 on dev, and dev loss was still falling at the last epoch. Equal train and
dev cosine with the loss still dropping is the signature of underfitting: the
model had not yet learned the vector-to-text map, so the corrector had nothing to
refine. vec2text's base sits
near **0.9** cosine before any correction, after 5M pairs on a GPU. Close that
gap and the whole pipeline moves, because the correction loop already works — it
added 0.09 cosine and behaved exactly as the paper describes (big first round,
convergence by round 3).

The single tracked metric to watch is `dev_cos` from `--eval-cos`. **Ship Tier 2
only if the zero-step base clears ~0.80 dev cosine.** Below that, the corrector
has too little to work with and the demo will still be a paraphraser.

## What to change, in order of expected payoff

1. **More pairs.** 40k → 1–3M. This is the dominant lever; the PoC never left the
   rising part of the loss curve. Source below.
2. **More epochs.** The PoC's 3 epochs were still improving. Train to a real dev-loss
   plateau, watching `dev_cos` rather than a fixed epoch count.
3. **A bigger base than t5-small (60M).** t5-base (220M) is the natural next rung;
   the M5 Max has the memory for it. Optional — do it only if t5-small plateaus
   below 0.80 dev cosine with ample data. It roughly triples export size.
4. **More soft tokens.** `--k 16` or `--k 32` gives the vector more room in the
   encoder input. Cheap to try; measure it.
5. **Longer corrector training.** The PoC ran the corrector for one epoch. With a
   stronger base, more corrector data and epochs pay off more.

Do not reach for 3 before exhausting 1 and 2. The finding was that the PoC was
data- and step-starved; capacity was not the limit.

## Environment (M5 Max)

```bash
git clone https://github.com/oaustegard/experiments && cd experiments/embedding-inversion
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers tokenizers onnxruntime onnx onnx_ir \
            numpy pyarrow sacrebleu huggingface_hub
```

PyTorch on Apple Silicon uses the `mps` backend; `train.py --device auto` picks it
up. `--bf16` turns on bf16 autocast — the M5 Max supports it and it roughly halves
memory. If an op ever fails on `mps` (rare on T5), run that step with
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

## 1. Build a big corpus

The generator is the same shape as the PoC (short questions + Wikipedia sentences,
so the length axis stays populated), just larger. The extra sentences come from
`sentence-transformers/wikipedia-en-sentences` (7.8M sentences, two parquet files,
~680 MB):

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
for i in (0, 1):
    hf_hub_download("sentence-transformers/wikipedia-en-sentences",
                    f"data/train-0000{i}-of-00002.parquet", repo_type="dataset", local_dir="data")
for repo, fn in [("sentence-transformers/natural-questions","pair/train-00000-of-00001.parquet"),
                 ("sentence-transformers/msmarco","queries/train-00000-of-00001.parquet")]:
    hf_hub_download(repo, fn, repo_type="dataset", local_dir="data")
PY

python3 build_data.py --n-train 2000000 --n-dev 2000 --n-test 2000 \
  --wiki-parquet data/data/train-00000-of-00002.parquet data/data/train-00001-of-00002.parquet
```

The embed pass is the slow part: bekko on CPU did ~500 texts/s in the container,
so 2M texts is a couple of hours. It runs once; `build_data.py` caches
`data/emb_*.npy`. The dedupe is now case-insensitive, closing `ERRORS.md` #5.

Keep the same test set discipline: split by string, so no test string appears in
training. `recheck.py` on the site data checks this.

## 2. Train the base, watching dev cosine

```bash
python3 train.py --mode zero --cond float --epochs 8 --bs 128 \
  --device mps --bf16 --eval-cos 500 --save-every 2000
```

`--eval-cos 500` greedy-decodes 500 dev items after each epoch and prints
`dev_cos` / `train_cos` / `dev_exact`. Stop when `dev_cos` plateaus. `--save-every
2000` checkpoints mid-epoch so a killed run resumes where it stopped (epochs are
long now). Watch that `dev_cos` and `train_cos` stay close — if `train_cos` pulls
ahead, you have started to overfit, and the fix is more data rather than more epochs.

If t5-small plateaus below ~0.80, switch the base once and only once:

```bash
# in model.py, T5_ID = "google-t5/t5-base"   (then retrain from scratch)
```

## 3. Corrector and evaluation, unchanged from the PoC

```bash
python3 evaluate.py gen  --cond float --split train        # zero-step hypotheses to train the corrector
python3 evaluate.py gen  --cond float --split dev
python3 train.py --mode correct --cond float --epochs 3 --bs 128 --device mps --bf16 \
  --hyps data/hyps_float_train.json --dev-hyps data/hyps_float_dev.json --init ckpt/zero_float.pt
python3 evaluate.py eval --cond float --rounds 5           # writes results_float.json
```

The `bin1` arm (the 1-bit-index condition) is optional for the demo; the float arm
is what a live page inverts. Run it the same way with `--cond bin1` if you want the
comparison to hold at the new scale.

## 4. Export to ONNX and verify

```bash
python3 export_onnx.py --cond float --verify 64
```

This writes, per model, an encoder and a decoder graph in three forms, and prints
a decode-agreement check against PyTorch. Sizes from the 40k model (they scale with
the base, not the data):

| graph | fp32 | 8-bit weights (`.w8`) | + int8 token tables (`.w8g`) |
|---|---|---|---|
| zero-step encoder | 94 MB | 38 MB | 38 MB |
| zero-step decoder | 233 MB | 109 MB | 60 MB |
| corrector encoder | 197 MB | 141 MB | 91 MB |
| corrector decoder | 233 MB | 109 MB | 60 MB |

The **`.w8g`** form is the one to ship: 8-bit MatMul weights (via `MatMulNBits`)
plus int8 token tables (via `Gather` quant). Both keep fp32 activations —
**this matters**: the ordinary `quantize_dynamic` int8 path quantizes activations
too and it *breaks* a T5 decoder into degenerate loops (0/32 on the PoC). The
verify step reports how many greedy decodes still match PyTorch exactly; on the PoC
the `.w8g` decoder held 10/16 with verifier cosine 0.591 vs 0.612. Re-check this on
the real model and drop to fp32 if the drop is too large.

Shipping budget for the live demo, float arm, `.w8g`, t5-small base:

- inverter: zero enc 38 + zero dec 60 + corrector enc 91 + corrector dec 60 = **~249 MB**
- bekko encoder (already served on the site): 124 MB
- **~370 MB total** behind a "load models" button. t5-base roughly triples the
  inverter half.

That is the honest number and it is heavier than a first guess. If it is too heavy:
ship the zero-step model only (98 MB + bekko) and drop the correction loop, or keep
the corrector but skip its separate encoder by conditioning it the same way as the
zero-step (a code change worth ~91 MB, not attempted here).

## 5. Ship it

Model files exceed the 100 MB git limit and this session cannot create releases
(the proxy 403s release creation), so from the Mac:

```bash
gh release create embedding-inversion-onnx-v1 onnx/*float*.w8g.onnx \
  --repo oaustegard/experiments --title "Inverter ONNX v1" --notes "float arm, w8g"
```

The browser page (a Tier 2 sibling of `ai-tools/embedding-inversion.html`) then
fetches the shards from the release, runs bekko to embed the typed sentence,
runs the zero-step beam, and loops the corrector with bekko as verifier —
the same algorithm `evaluate.py` runs, ported to JS with `onnxruntime-web`.
Pin the WASM backend, not WebGPU: transformers.js int8 on WebGPU silently
collapses bekko's embedding space (`METHODS.md`, and `taxonomy-snap.html`
learned it the hard way).

## The decision gate

Everything above is worth doing only if step 2 clears **~0.80 dev cosine**. If it
does, Tier 2 is a real text-recovery demo and the page is a straightforward port of
the evaluation loop. If it does not, the finding stands — bekko's 384 dims do not
give back enough to reconstruct arbitrary text — and the honest demo is the Tier 1
explorer already live, plus this document as the record of what the ceiling was and
what it would take to raise it.
