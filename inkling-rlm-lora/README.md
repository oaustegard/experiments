# inkling-rlm-lora

A recipe for teaching Inkling 2 Mini to work as a recursive language model (RLM) with a
LoRA adapter, on hardware we can afford. The method comes from Zhang, Chakraborty and
Khattab, *Language Model Harnesses Can Be Compositional Generalizers* (alphaXiv
2611.harness-cg, 20 Nov 2026). They fine-tuned Qwen3.8-27B into RLM-27B, which scores 58.5
against the base model's 46.1 on eight held-out long-context benchmarks run in the same
harness and uses about 1.9 times fewer tokens.

**Status: recipe, unrun.** The training script's interrupt-and-resume path is tested on a
tiny model (see [Mac track](#mac-track-mlx-interruptible)). The recipe has not touched
Inkling yet.

## What is already published

| Item | Where | What it gives us |
|---|---|---|
| Training and evaluation code | [alexzhang13/harness-cg](https://github.com/alexzhang13/harness-cg/) | the RLM harness (Python REPL, `llm_query` sub-calls, depth 1) and its evaluation |
| SFT traces | [`a1zhang/synthetic-rlm-sft-data`](https://hf.co/datasets/a1zhang/synthetic-rlm-sft-data) | 9,480 traces, one config per benchmark family, synthetic stand-ins with no real benchmark rows |
| Their model | [`a1zhang/RLM-27B`](https://hf.co/a1zhang/RLM-27B) | full fine-tune, 1 epoch, AdamW, lr 5e-6 cosine, 32,768-token packs, 8×H100 |
| Inkling on Transformers and PEFT | [`idoru/inkling-2-mini-harness-adapters`](https://hf.co/spaces/idoru/inkling-2-mini-harness-adapters) | a Transformers backport of Inkling whose logits and gradients exactly match the native release; PEFT rank-32 LoRA on the attention projections and every expert, loadable in vLLM 0.25.1 |
| Inkling on MLX | [`mlx-community/Inkling-2-Mini-4bit`](https://hf.co/mlx-community/Inkling-2-Mini-4bit) | needs mlx-lm [PR #2086](https://github.com/ml-explore/mlx-lm/pull/2086); keeps the experts in their native NVFP4 (layers 0–19) and MXFP8 (20–23) |

The `idoru` Space is a different experiment: GRPO on the Pi, Hermes and OpenCode coding
harnesses, after Hugging Face's harness-adapters work. Its porting work carries over. Two
of its findings matter here. Stock TRL `AsyncGRPOTrainer` rejects Inkling's layer types
(SkyRL has an [Inkling2 recipe](https://docs.skyrl.ai/docs/models/inkling2)). After one
GRPO update its pilot was worse than the frozen model, 5 of 16 Terminal-Bench tasks
against 8 (p = 0.25).

## Facts the recipe rests on

- **Inkling's weights are 4-bit.** `config.json` has `quant_algo: NVFP4` (modelopt);
  the Hub's "8-bit" tag is misleading. Training a bf16 adapter on frozen 4-bit weights
  then trains against the weights that will be served, which QLoRA on a bf16-trained
  model cannot do. NVFP4 runs natively only on Blackwell GPUs.
- **Root traces are short.** On `loogle` (397 traces) and `longreason` (1,821), total
  message length has a median of 13.3K and 10.3K characters, p90 19.3K and 18.2K, max
  26K and 41K. At about 4 characters per token that is a median near 3K tokens and a p90
  near 5K. These counts are from characters; the tokenizer was not run (see
  [Not yet checked](#not-yet-checked)). The harness keeps the long document in the REPL,
  so the model's own context stays small, and so does training memory.
- **The traces probably come from Qwen, not Opus.** The one row inspected field by field
  has `teacher_model: qwen38-27b`, hint mode `note`: the base model run with a hint
  program. Every row in `loogle` and `longreason` has reward 1.0. Count the
  `teacher_model` values before relying on this. Training Inkling on them is cross-model distillation. Running Inkling
  itself with the released hint programs (self-distillation) is the closer analogue to the
  paper, and comes second here.
- **Inkling's size.** 26.4B parameters, 24 layers, hidden size 2304, 128 routed experts
  with 8 active, a looped core (layers 6–17 run three times). By my arithmetic that is
  about 4.4B parameter-equivalents of compute per token, loop passes and output head
  included.

## Plan

Each stage ends at a stop condition.

0. **Desk checks.** Done 2026-11-22. Their code and data are released, so no trace
   generation is needed.
1. **Baseline, inference only.** Frozen Inkling in the `harness-cg` harness, with Inkling
   also serving the sub-calls, on 20–30 items from each of `oolong_real`, `longreason`
   and `loogle`. Run it on Kaggle (Q4 GGUF, llama.cpp, two free T4s) or on the Mac with
   `mlx_lm.server`. Evaluation is the expensive part: the paper's harness averaged 385K
   tokens per task for RLM-27B and 721K for the base model. **Stop** if the frozen model
   already uses the harness well, because there is then little for an adapter to add.
2. **Data.** 170 traces from each of the same three families (510 in all), 10 per family
   held out for validation loss. `train_mlx.py prep` does this.
3. **Rank sweep, SFT.** Ranks 1, 4 and 16 on the same data and the same held-out items.
   - Scale is alpha/rank with alpha fixed at 32, so the best learning rate should not
     move with rank.
   - The learning rate is 5e-5, ten times the paper's full fine-tuning rate, following
     Thinking Machines' "LoRA Without Regret" (2025).
   - That post found rank 1 enough for RL, where an episode carries about one bit. SFT
     carries more, but 510 traces is a small dataset.
   - **Read** rank 1 within noise of rank 16 as a low-dimensional change, which points at
     rank-1 GRPO. A score that climbs with rank means SFT is teaching capability, not
     format.
   - **Stop** if no rank beats the baseline by more than seed-to-seed noise.
4. **Rank-1 GRPO, only if stage 3 gains.** Short tasks for training, longer ones for
   evaluation, as the paper did on Qwen3-30B-A3B. Use SkyRL, not TRL. Start with a small
   learning rate, given the `idoru` pilot's first update.

## Hardware tracks

| Track | Hardware | Cost | Use for |
|---|---|---|---|
| Kaggle | 2× T4 (15 GB each), 30 GPU-hours a week free | 0 | stage 1 inference only; T4s support neither bf16 nor NVFP4 |
| Mac (MLX) | Oskar's M5 Max, 128 GB | 0, overnight | stages 2–3, interruptible |
| Colab | A100, H100, G4 | compute units | stage 3 if the Mac is too slow; stage 4 rollouts with vLLM serving the adapter |
| Tinker | Thinking Machines' service | credits, if granted | stage 4; may support Inkling natively (unverified) |

On Colab, check whether G4 is an RTX PRO 6000 (Blackwell) before relying on it. It is the
only Colab GPU that would run NVFP4 natively, and the docs do not say. On an A100 or H100
the NVFP4 weights are emulated or upcast.

## Mac track (MLX, interruptible)

[`train_mlx.py`](train_mlx.py) trains a LoRA adapter on the MLX conversion and can be
stopped at any time.

**Interrupting.** Ctrl-C or SIGTERM finishes the current optimizer step, saves, and exits.
A second Ctrl-C exits at once and loses at most `--save-every` optimizer steps. Rerunning
the same command resumes from the last checkpoint.

**What a checkpoint holds.** Each `ckpt-<step>/` holds the adapter weights, the AdamW
state (moments, step count, learning-rate schedule position) and the data position. A
`LATEST` pointer is swapped atomically once the directory is complete, so a crash during a
save leaves the previous checkpoint in force. The last two are kept.

**Resume test.** `python train_mlx.py selftest` builds a 4-layer Qwen3-MoE with experts
quantized to NVFP4, as in the Inkling conversion. It trains 12 steps uninterrupted, then
trains again with a SIGINT mid-run and a resume. On 2026-11-22 on Linux CPU (MLX 0.32.4,
mlx-lm 0.32.2), the interrupted run stopped at step 5, and both runs ended with
bit-identical adapters: max absolute difference 0.0 over 56 tensors.

### Running it

```bash
python3 -m venv ~/.venvs/inkling && . ~/.venvs/inkling/bin/activate
pip install -U mlx transformers huggingface_hub \
  "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/2086/head"

python train_mlx.py selftest                      # about a minute
python train_mlx.py prep --out data               # 510 train + 30 val traces
python train_mlx.py train --dry-run --rank 1      # lists adapted modules and trainable params
python train_mlx.py train --rank 1 --max-steps 3 --out runs/smoke   # first real numbers

caffeinate -i python train_mlx.py train --rank 1 --out runs/r1
caffeinate -i python train_mlx.py train --rank 4 --out runs/r4
caffeinate -i python train_mlx.py train --rank 16 --out runs/r16
```

**Running and evaluating.**
- `caffeinate -i` keeps the Mac from sleeping, which would otherwise stall the run.
- Progress goes to stdout and to `runs/<r>/log.jsonl`: loss, validation loss every 10
  steps, tokens per second and ETA.
- The finished adapter is `runs/<r>/adapters.safetensors` plus an mlx-lm
  `adapter_config.json`. `mlx_lm.server --model mlx-community/Inkling-2-Mini-4bit
  --adapter-path runs/r1` then serves it to the harness for evaluation.

**Defaults.**
- Adapters go on every linear and expert projection in `model.layers`, except the router
  and the loop gates (`--exclude`).
- An adapter on a looped core layer is shared by all three passes, just as that layer's
  weights are.
- Batch size 1, 8 accumulated steps per update, 1 epoch, 5% warmup then cosine decay,
  gradient checkpointing on, sequences cut at 8,192 tokens.

### How slow

These figures are an estimate, not a measurement. Training costs about three forward
passes, which is roughly 26 GFLOP per token at 4.4B parameter-equivalents. MLX's
quantized expert kernels are unlikely to sustain more than 3–10 TFLOPS on an M5 Max,
which gives 100–400 tokens a second. 510 traces at about 3.5K tokens is 1.8M tokens per
epoch, so **one rank takes roughly 1.2–5 hours** and the three-rank sweep fits in one or
two nights. The `--max-steps 3` smoke run prints the real tokens per second; replace this
paragraph with it.

Memory should not bind on 128 GB. The weights are about 15–17 GB (NVFP4 experts, 4-bit
attention, a bf16 vision tower), and 8K-token sequences with gradient checkpointing add
little. A 32 GB Mac may manage with `--max-tokens 4096`.

## Not yet checked

- **`prep` on Inkling's tokenizer.** The prefix-diff masking assumes Inkling's chat
  template renders each prefix of a conversation as a prefix of the next. A trace where
  it does not is dropped and counted, so a high drop count means the template merges
  turns. This test was blocked in the authoring session (it downloads the tokenizer from
  the Hub).
- **What `train_from_turn` counts.** The script reads it as the index of the first
  assistant turn to train on, with -1 meaning all. The one row inspected has -1.
- **PR #2086's model structure.** It must expose `model.layers` with the looped core
  inside it, and backward must work through the loop gates and the NVFP4 expert kernels
  on Metal. The selftest covers NVFP4 experts on CPU only. `train --dry-run` and the
  3-step smoke run will show both.
- **Sub-calls in the harness.** Whether `harness-cg`'s sub-calls should be served by the
  adapted model or the frozen one. The paper's surrogate training used fake sub-calls,
  which argues for the frozen base.
