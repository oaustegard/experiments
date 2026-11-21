# monad-looped

PleIAs' [Monad](https://huggingface.co/PleIAs/Monad) is a 56.7M-parameter reasoning model, 64 layers deep and 256 wide,
that writes its reasoning inside a `<think>` block before answering. Thinking Machines' Inkling 2 Mini (2026-11-20) runs
its middle layers three times with shared weights and reports reaching answers with fewer reasoning tokens. The question
here: can Monad be made deeper by looping a band of its own layers, and can that latent depth replace enough written
reasoning to make answers faster? The plan and its predictions were registered before any arm ran: [`PLAN.md`](PLAN.md).

## Answer

No, not at this scale and budget. The loop reruns a band of layers up to three times; each extra pass reads the band's
input plus a gated share ("mixing") of the previous pass's output, and gate 0 is the plain model. Quality is answer NLL,
nats per token on reference answers from held-out SYNTH rows (Monad's training set).

Without training, every looped setting is worse than the base model: the best adds 0.002 nats <!-- @evidence[[results/summary.json#delta.best_screen_f1]] -->,
full mixing at 3 passes adds 0.556 <!-- @evidence[[results/summary.json#delta.B1_K3_a1p0_f1]] -->. Retrofitted with 300M
tokens and a learned gate starting at zero, the gate stayed near 0.013 <!-- @evidence[[results/summary.json#gate.loopB1_final_mean]] -->,
so extra passes changed nothing and the arm matched a same-data control without a loop to 0.001 nats. With the gate forced
open on a schedule, training drove a learned correction to −0.92 <!-- @evidence[[results/summary.json#gate.loopB1r_last_mean]] -->
against the scheduled 1.0, closing it again; at step 6,000 that arm was level with the control (1.1118 against 1.1116).

The speedup stage (training shorter traces) was not started. What would change this: a far larger training budget with the
gate held open, or a model wide enough that a repeated pass has room to compute something new.

## Findings

1. **Training-free looping only hurts, in proportion to how much of the previous pass is mixed in.** On 384 held-out SYNTH
   rows, base answer NLL is 1.1046 with the full trace <!-- @evidence[[results/summary.json#base.f1]] -->; the
   gentlest arm (layers 12–19, 2 passes, mixing 0.1) gives 1.1068 <!-- @evidence[[results/summary.json#screen.B1_K2_a0p1.f1]] -->,
   and the 16-layer band at 3 passes with full mixing 1.9826 <!-- @evidence[[results/summary.json#screen.B3_K3_a1p0.f1]] -->.
   The ordering holds for every band, pass count and trace fraction (Round 2).
2. **A learned gate that starts at zero stays shut.** After 300M tokens the gate on the looped band averaged 0.013, largest
   entry 0.034 <!-- @evidence[[results/summary.json#gate.loopB1_final_absmax]] -->. Final answer NLL at 1 and 3 passes:
   1.1057 and 1.1057 <!-- @evidence[[results/summary.json#loopB1_K3_final.f1]] -->, control 1.1056 <!-- @evidence[[results/summary.json#control_final.f1]] --> (Round 3).
3. **A forced-open gate is undone by training.** With a scheduled gain ramping to 1.0 over the first 30% of training plus a
   learned per-channel correction, the correction reached −0.61 by step 3,300 (scheduled gain 0.88) and −0.92 by step 7,300,
   leaving an effective gate near 0.08 (1.0 − 0.92). At step 6,000: answer NLL 1.1118 at 2 passes <!-- @evidence[[results/summary.json#loopB1r_K2_6000.f1]] -->
   against the control's 1.1116 <!-- @evidence[[results/summary.json#control_6000.f1]] -->; with no trace kept,
   1.4466 against 1.4421 (Round 4).
4. **No sign of reasoning moving into the latent.** The prediction (PLAN.md P4) was that a used loop would help most when
   the trace is truncated. At step 6,000 the forced-gate arm at 2 passes trails the control by 0.0045 nats with no trace kept
   and by 0.0002 with the full trace, so truncation makes it slightly worse, not better; the learned-gate arm's gaps are
   within ±0.003 at every fraction (Rounds 3–4).
5. **Continued training on SYNTH does not improve Monad on held-out SYNTH.** The control's final answer NLL is 1.1056 against
   the base model's 1.1046, so 300M more tokens of Monad's own data at learning rate 1e-4 left it where it was (Round 3).
6. **Dev-set accuracy is too noisy at this size to separate arms.** On the 128 MMLU + GSM8K prompts of the 192-prompt dev
   fixture (a different, harder-to-score set than Finding 7's test prompts), base Monad answers
   9.4% <!-- @evidence[[results/summary.json#dev.base.acc]] -->, the control 13.3% and the screen arms 0–14.8%; a few
   prompts move these numbers. Full mixing at 3 passes makes 64.8% of answers end in a repetition loop <!-- @evidence[[results/summary.json#dev.B1_K3_a1p0.loop]] --> (Round 2).
7. **Baseline cost of a right answer.** On the 350 MMLU + GSM8K test prompts from `monad-weight-edit`, the loop-edited
   model is right 18.9% of the time <!-- @evidence[[results/summary.json#baseline.A8p_acc]] -->, thinks for 520 tokens on
   average when it closes its think block <!-- @evidence[[results/summary.json#baseline.A8p_think]] -->, and generates
   3,357 tokens per correct answer <!-- @evidence[[results/summary.json#baseline.A8p_tok_per_correct]] --> (Round 1).

## Method

**Model and loop** (`loopmodel.py`). Monad reimplemented in plain PyTorch, loading `model.safetensors` directly. Layers
`[0, s)` are the prelude, `[s, e)` the band, `[e, 64)` the coda. Pass 1 of the band reads the prelude output x_p; pass k
reads `x_p + g_k ⊙ (h_{k−1} − x_p)`, with h the previous pass's band output and g a per-channel gate. Each pass keeps its own
KV cache. g = 0 reproduces the base model; `check_parity.py` matches Hugging Face's Monad to 0.0 logit difference and cached
decoding to uncached decoding token for token. Inkling mixes through attention residuals; the gated interpolation is a
simplification.

**Bands.** B1 = layers 12–19 and B2 = 36–43 (8 layers each, low block influence and deletion KL in
`monad-weight-edit/results/layers.json`, clear of the attention-sink and loop layers 22–34), B3 = 6–21 (16 layers).

**Teacher-forced metrics** (`tf_eval.py`, `data/eval_synth.jsonl`). 384 English SYNTH rows from shard 500, which no training
run read: 128 each of memorization, math exercise and mcq. Answer NLL per token with 0%, 50% or 100% of the reference trace
kept; trace NLL; KL to the first config.

**Generation** (`gen_eval.py`, `score.py`). Greedy decoding on `monad-weight-edit`'s 192-prompt dev fixture, 1,024 new
tokens (1,536 for IFEval). Scored on MMLU and GSM8K only: IFEval's checker needs NLTK data, whose download was refused in
this environment.

**Training** (`prep_data.py`, `train.py`). SYNTH shards 1–5, English rows, 607k rows and 614M tokens; each row is the chat
prompt then `reasoning </think> answer`, loss on the target. Continued from base Monad: AdamW, learning rate 1e-4 cosine with
2% warmup, gate learning rate 1e-3, 24,576-token length-bucketed batches, bf16 autocast, torch.compile per layer, 300M tokens
(12,518 steps). Every step draws its pass count from {1, 2, 3} in a 1:2:1 ratio; the control draws the same sequence and
ignores it, so both see identical batches in identical order. Two Colab A100-40GB VMs; the control trained at 142,677 tokens/s <!-- @evidence[[results/summary.json#throughput.a100_control_tok_s]] -->, the looped arm at 129,164.

**Arms.** control (no loop); loopB1 (B1, learned gate from zero); loopB1r (B1, learned gate plus a scheduled gain rising from
0 to 1 over the first 30% of steps; stopped at step 7,300 when the experiment was cut, scored at its step-6,000 checkpoint).

## Log

### Round 1 — baseline (2026-11-21)

Reasoning length and accuracy from `monad-weight-edit`'s Mac GPU test runs, rescored for think length. Base Monad 16.9% right
on MMLU + GSM8K, 4,290 tokens per correct answer; loop-edited A8p 18.9%, 3,357. MMLU sits at chance (27% for A8p), so
generation accuracy on this fixture can only show large changes; the teacher-forced SYNTH metric was added for sensitivity.

### Round 2 — training-free screen

25 configs (3 bands × 2 or 3 passes × mixing 0.1/0.25/0.5/1, plus base). The Kaggle T4 job sat queued for over three hours;
the screen ran on a Colab A100 instead. P1 (no arm beats base; best within 0.02 nats; mixing 1 worse by > 0.1 for every
band) held: best +0.0022, mixing 1 between +0.125 and +0.878. P2 (no accuracy gain beyond noise; mixing ≥ 0.5 raises loops)
held in direction; only 10 of the 25 generation configs finished before the cut.

### Round 3 — retrofit with a learned gate

The first attempt on two Colab L4s was lost: both VMs were reclaimed about 24 minutes after the last `colab exec`, although
the watcher downloaded each log every 4 minutes. Relaunched on two A100s with an exec keepalive and checkpoint mirroring.
Saved checkpoints from the compiled model carried `_orig_mod.` in their keys and failed to load until the loader stripped
it. Result: the gate stayed at about 0.01 and the looped arm tracked the control at every checkpoint (Findings 2, 5). P3
(the looped arm beats the control at 3 passes) failed.

### Round 4 — retrofit with a scheduled gate (added after Round 3)

A design change made after seeing Round 3: Inkling's write-up says its gate "increases the gain on the message over time",
which reads as a schedule. The scheduled gain reached 1.0 at step 3,755, but the learned correction cancelled it (Finding 3).
Cut at step 7,300 when Oskar ended the experiment at 1 PM EST; the step-6,000 checkpoint was scored against the control at
the same step. P4 (the loop helps most with a truncated trace) failed with the rest.

Compute: the Colab balance went from 321.9 to 292.0 units over the day (the A100 pair and the lost L4 round); no Kaggle quota used.
