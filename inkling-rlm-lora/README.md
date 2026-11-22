# inkling-rlm-lora

A recipe for training a LoRA adapter that teaches Inkling 2 Mini (Thinking Machines'
26.4B-parameter open mixture-of-experts model, released 17 Nov 2026) to work as a
recursive language model, on hardware we can afford.

A recursive language model (RLM) never reads a long document directly. The document sits
in a Python REPL. The model writes code to inspect it and calls a sub-model on pieces
through an `llm_query` function, then submits an answer. The conversation between the
top-level model and the REPL is the **root trace**. The sub-calls happen outside it.

The method comes from Zhang, Chakraborty and Khattab, *Language Model Harnesses Can Be
Compositional Generalizers* (alphaXiv 2611.harness-cg, 20 Nov 2026), called harness-cg
below after its code repository. They fine-tuned Qwen3.8-27B into RLM-27B. Both were
evaluated in the RLM harness on eight held-out long-context benchmarks. RLM-27B scored
58.5 against Qwen3.8-27B's 46.1 and used about 1.9 times fewer tokens.

**Status: recipe, unrun.** The training script's interrupt-and-resume path is tested on a
tiny model (see [Mac track](#mac-track-mlx-interruptible)). Nothing has been run on
Inkling yet.

## What is already published

| Item | Where | What it gives us |
|---|---|---|
| Training and evaluation code | [alexzhang13/harness-cg](https://github.com/alexzhang13/harness-cg/) | the RLM harness (depth 1: the sub-model does not recurse further) and its evaluation |
| SFT traces | [`a1zhang/synthetic-rlm-sft-data`](https://hf.co/datasets/a1zhang/synthetic-rlm-sft-data) | 9,480 root traces, one config per benchmark family, on synthetic tasks with no real benchmark rows |
| The paper's model | [`a1zhang/RLM-27B`](https://hf.co/a1zhang/RLM-27B) | full fine-tune, 1 epoch, AdamW, lr 5e-6 cosine, 32,768-token packs, 8×H100 |
| Inkling on Transformers and PEFT | [`idoru/inkling-2-mini-harness-adapters`](https://hf.co/spaces/idoru/inkling-2-mini-harness-adapters) | a Transformers port of Inkling whose logits and gradients exactly match the native release; PEFT rank-32 LoRA on the attention projections and every expert, loadable in vLLM 0.25.1 |
| Inkling on MLX | [`mlx-community/Inkling-2-Mini-4bit`](https://hf.co/mlx-community/Inkling-2-Mini-4bit) | needs mlx-lm [PR #2086](https://github.com/ml-explore/mlx-lm/pull/2086); keeps the experts in their native NVFP4 (layers 0–19) and MXFP8 (20–23) |

The `idoru` Space runs reinforcement learning (GRPO) on Inkling in three coding-agent
harnesses, not the RLM harness. Two of its findings apply to stage 4 below. Stock TRL
`AsyncGRPOTrainer` rejects Inkling's layer types, while SkyRL has an
[Inkling2 recipe](https://docs.skyrl.ai/docs/models/inkling2). And after one GRPO update
its pilot solved 5 of 16 Terminal-Bench tasks against 8 for the frozen model (exact
McNemar p = 0.25).

## Facts the recipe rests on

- **Inkling's weights are 4-bit.** Its `config.json` has `quant_algo: NVFP4` (NVIDIA's
  4-bit float format, written by modelopt), although the model's Hub page carries an
  "8-bit" tag. A bf16 adapter trained on the frozen 4-bit weights therefore trains
  against the weights that will be served, unlike QLoRA on a model trained in bf16.
  NVFP4 runs natively only on Blackwell GPUs.
- **Root traces are short.** On `loogle` (397 traces) and `longreason` (1,821), total
  message length has a median of 13.3K and 10.3K characters, p90 19.3K and 18.2K, max
  26K and 41K. At about 4 characters per token that is a median near 3K tokens and a p90
  near 5K. These are character counts; Inkling's tokenizer was not run on them. Training
  memory stays small because the long document lives in the REPL, not in the root trace.
- **The traces were probably written by Qwen.** The one row inspected field by field has
  `teacher_model: qwen38-27b` and hint mode `note`. That is, the base Qwen model solved
  the task with a reference solution program in its prompt as a hint. Every row in
  `loogle` and `longreason` has reward 1.0. Training Inkling on these traces is
  distillation from another model. Count the `teacher_model` values before relying on
  this.
- **Inkling's shape.** 24 layers, hidden size 2304, 128 routed experts with 8 active per
  token, and a looped core: layers 6–17 run three times per token. Counting the loop
  passes and the output head, that is about 4.4B parameter-equivalents of compute per
  token (my arithmetic from `config.json`).

## Plan

Each stage ends at a stop condition. Stages 1 and 3 are scored the same way: the
`harness-cg` evaluation on a fixed set of benchmark items, compared item by item.
Validation loss on held-out traces is a training diagnostic, not the result.

1. **Baseline, inference only.** Frozen Inkling in the `harness-cg` harness, with Inkling
   also serving the sub-calls, on 20–30 items from each of `oolong_real`, `longreason`
   and `loogle`. Run it on Kaggle (Q4 GGUF, llama.cpp, two free T4s) or on the Mac with
   `mlx_lm.server`. Evaluation costs far more than training: the paper reports an
   average of 385K tokens per task for RLM-27B and 721K for the base model, counted over
   every model call, sub-calls included. Run the baseline twice to measure run-to-run
   noise. **Stop** if the frozen model already scores near RLM-27B's level on these
   items, because there is then little for an adapter to add.
2. **Data.** 170 traces from each of the same three families (510 in all), plus 10 per
   family held out for validation loss. `train_mlx.py prep` does this.
3. **Rank sweep, SFT.** Ranks 1, 4 and 16 on the same data, scored on the stage 1 items.
   - Scale is alpha/rank with alpha fixed at 32, so the best learning rate should not
     move with rank.
   - The learning rate is 5e-5, ten times the paper's full fine-tuning rate, following
     Thinking Machines' "LoRA Without Regret" (2025).
   - That post found rank 1 enough for RL, where an episode carries about one bit. SFT
     carries more, but 510 traces is a small dataset.
   - **Read** rank 1 within noise of rank 16 as a low-dimensional change, which points at
     rank-1 GRPO. A score that climbs with rank means SFT is teaching capability as well
     as format.
   - **Stop** if no rank beats the baseline by more than the run-to-run noise from
     stage 1.
4. **Rank-1 GRPO, only if stage 3 gains.** Train on short tasks and evaluate on longer
   ones, as the paper's length-generalization runs did on Qwen3-30B-A3B. Use SkyRL, not
   TRL. Start with a small learning rate, given the `idoru` pilot's first update.

A second data option for stage 2: run Inkling itself in the harness with the released
hint programs and keep its correct traces. That is closer to how the paper made its data,
and worth trying if the Qwen traces underperform.

## Hardware tracks

| Track | Hardware | Cost | Use for |
|---|---|---|---|
| Kaggle | 2× T4 (15 GB each), 30 GPU-hours a week free | 0 | stage 1 inference only; T4s support neither bf16 nor NVFP4 |
| Mac (MLX) | Oskar's M5 Max, 128 GB | 0, overnight | stages 2–3, interruptible |
| Colab | A100, H100, G4 | compute units | stage 3 if the Mac is too slow; stage 4 rollouts with vLLM serving the adapter |
| Tinker | Thinking Machines' training service | credits, if granted | stage 4; may support Inkling natively (unverified) |

On Colab, check whether G4 is an RTX PRO 6000 (Blackwell) before relying on it. It would
be the only Colab GPU that runs NVFP4 natively, and the Colab docs do not say. On an A100
or H100 the NVFP4 weights are emulated or upcast.

## Mac track (MLX, interruptible)

[`train_mlx.py`](train_mlx.py) trains a LoRA adapter on the MLX conversion and can be
stopped at any time.

**Interrupting.** Ctrl-C or SIGTERM finishes the current optimizer step, saves, and exits.
A second Ctrl-C exits at once and loses at most `--save-every` optimizer steps (default
5). Rerunning the same command resumes from the last checkpoint.

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
python train_mlx.py prep --out data               # writes data/train.jsonl and data/val.jsonl
python train_mlx.py train --dry-run --rank 1      # lists adapted modules and trainable params
python train_mlx.py train --rank 1 --max-steps 3 --out runs/smoke   # measures tokens per second

caffeinate -i python train_mlx.py train --rank 1 --out runs/r1
caffeinate -i python train_mlx.py train --rank 4 --out runs/r4
caffeinate -i python train_mlx.py train --rank 16 --out runs/r16
```

- `prep` tokenizes each trace and masks it so that only the model's own replies are
  trained on. It finds the replies by rendering the chat template on each successive
  prefix of the conversation and taking the difference.
- `train` reads `data/train.jsonl` and `data/val.jsonl` by default (`--data`, `--val`).
- `caffeinate -i` keeps the Mac from sleeping, which would otherwise stall the run.
- Progress goes to stdout and to `runs/<r>/log.jsonl`: loss, validation loss every 10
  steps, tokens per second and ETA.
- The finished adapter is `runs/<r>/adapters.safetensors` plus an mlx-lm
  `adapter_config.json`.

**Evaluating.** `mlx_lm.server --model mlx-community/Inkling-2-Mini-4bit --adapter-path
runs/r1` serves the adapted model on an OpenAI-compatible endpoint. Point the
`harness-cg` evaluation at that endpoint for the root model, and at the server without
`--adapter-path` for sub-calls (see [Not yet checked](#not-yet-checked)). I have not read
`harness-cg`'s evaluation entry point, so the exact command is still to be written.

**Defaults.**
- Adapters go on every linear and expert projection in `model.layers`, except the router
  and the loop gates. `--exclude` takes a regular expression over module paths and
  replaces that exclusion list.
- An adapter on a looped core layer is shared by all three passes, just as that layer's
  weights are.
- Batch size 1, 8 accumulated steps per update (`--accum`), 1 epoch, 5% warmup then
  cosine decay, gradient checkpointing on, sequences cut at 8,192 tokens
  (`--max-tokens`).

### How slow

These figures are an estimate, not a measurement. Training costs about three forward
passes, which is roughly 26 GFLOP per token at 4.4B parameter-equivalents. MLX's
quantized expert kernels are unlikely to sustain more than 3–10 TFLOPS on an M5 Max,
which gives 100–400 tokens a second. 510 traces at a guessed mean of 3.5K tokens (the
median is near 3K, with a long tail) is 1.8M tokens per epoch, so **one rank takes
roughly 1.2–5 hours** and the three-rank sweep fits in one or two nights. The 3-step
smoke run gives the measured rate.

Memory should not bind on 128 GB. The weights are about 15–17 GB (NVFP4 experts, 4-bit
attention, a bf16 vision tower), and 8K-token sequences with gradient checkpointing add
little. A 32 GB Mac may manage with `--max-tokens 4096`.

## Not yet checked

- **`prep` on Inkling's tokenizer.** The reply masking assumes Inkling's chat template
  renders each prefix of a conversation as a prefix of the next rendering. A trace where
  it does not is dropped and counted in `prep`'s output, so a high drop count means the
  template merges turns.
- **What `train_from_turn` counts.** The script reads it as the index of the first
  assistant turn to train on, with -1 meaning all. The one row inspected has -1.
- **PR #2086's model structure.** It must expose `model.layers` with the looped core
  inside it, and backward must work through the loop gates and the NVFP4 expert kernels
  on Metal. The selftest covers NVFP4 experts on CPU only. `train --dry-run` and the
  3-step smoke run will show both.
- **Which model serves sub-calls.** The paper trained its root model against simulated
  sub-calls, which suggests serving sub-calls from the frozen base during evaluation.
