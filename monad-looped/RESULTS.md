# monad-looped

Inkling 2 Mini (Thinking Machines, 2026-11-20) runs its middle layers three times per token and needs fewer reasoning
tokens to reach an answer. Monad (PleIAs, 56.7M parameters, 64 thin layers) spends about 520 tokens thinking before each
answer. If Monad could do some of that thinking by rerunning a band of its layers instead of writing it out, it would answer
faster. This experiment tried to teach it that, and Monad refused every time.

## Answer

Monad will not use a repeated pass of its own layers, so the speedup is not available this way.

The loop reruns a band of layers (12–19 in the main arms) up to three times; each extra pass reads the band's input plus a
gated share of what the previous pass produced. Gate 0 is the unchanged model. Quality is answer NLL: how surprised the
model is by the reference answers of 384 held-out SYNTH questions (SYNTH is Monad's training set).

Three tries, one outcome:

- **Bolted on without training**, every setting made answers worse, from +0.002 nats at the gentlest <!-- @evidence[[results/summary.json#delta.best_screen_f1]] -->
  to +0.556 with full mixing at 3 passes <!-- @evidence[[results/summary.json#delta.B1_K3_a1p0_f1]] -->.
- **Trained in with a gate free to open** (300M tokens), the gate stayed at 0.013 <!-- @evidence[[results/summary.json#gate.loopB1_final_mean]] -->.
  The model learned to ignore the loop and ended level with a control trained on the same data without one.
- **Forced open on a schedule**, training grew a counter-weight of −0.92 against the +1.0 <!-- @evidence[[results/summary.json#gate.loopB1r_last_mean]] -->
  and shut it again.

So the loop never carried any reasoning, and the second stage (training shorter traces to cash in the speed) was not run.
A much larger training budget with the gate locked open, or a wider model, could change this.

## Findings

1. **Untrained, more looping is strictly worse.** Base answer NLL 1.1046 <!-- @evidence[[results/summary.json#base.f1]] -->;
   layers 12–19 rerun twice at gate 0.1 give 1.1068 <!-- @evidence[[results/summary.json#screen.B1_K2_a0p1.f1]] -->; the
   16-layer band rerun three times at gate 1 gives 1.9826 <!-- @evidence[[results/summary.json#screen.B3_K3_a1p0.f1]] -->,
   and 64.8% of its answers end in a repetition loop <!-- @evidence[[results/summary.json#dev.B1_K3_a1p0.loop]] -->. Every
   band, pass count and gate value orders the same way (Round 2).
2. **Given the choice, training keeps the loop closed.** The learned gate ended at 0.013 on average, 0.034 at most <!-- @evidence[[results/summary.json#gate.loopB1_final_absmax]] -->;
   one pass and three passes score 1.1057 <!-- @evidence[[results/summary.json#loopB1_K3_final.f1]] -->,
   the no-loop control 1.1056 <!-- @evidence[[results/summary.json#control_final.f1]] --> (Round 3).
3. **Forced open, training closes it anyway.** The learned counter-weight reached −0.61 by step 3,300 and −0.92 by step
   7,300, leaving about 8% of the message. At step 6,000 the arm scores 1.1118 <!-- @evidence[[results/summary.json#loopB1r_K2_6000.f1]] -->
   against the control's 1.1116 <!-- @evidence[[results/summary.json#control_6000.f1]] -->, and it is no better when the
   reference reasoning is cut out, where latent reasoning would have shown: 1.4466 against 1.4421 (Round 4).
4. **The extra training itself bought nothing.** 300M more tokens of SYNTH left the control at 1.1056 against the original
   model's 1.1046, so the loop arms were not competing with a moving target (Round 3).
5. **The bar a speedup had to clear.** The loop-edited Monad from `monad-weight-edit` is right 18.9% of the time <!-- @evidence[[results/summary.json#baseline.A8p_acc]] -->
   on 350 MMLU + GSM8K test prompts and writes 3,357 tokens per correct answer <!-- @evidence[[results/summary.json#baseline.A8p_tok_per_correct]] --> (Round 1).

## Method

**Model and loop** (`loopmodel.py`). Monad reimplemented in plain PyTorch, loading `model.safetensors` directly. Layers
`[0, s)` are the prelude, `[s, e)` the band, `[e, 64)` the coda. Pass 1 of the band reads the prelude output x_p; pass k
reads `x_p + g_k ⊙ (h_{k−1} − x_p)`, with h the previous pass's band output and g a per-channel gate. In the scheduled arm, g = s(t) + c: a gain s rising from 0 to 1 over the first 30% of training plus a learned per-channel correction c (the "learned correction" below). The coda reads the last pass's output. Each pass keeps its own
KV cache. g = 0 reproduces the base model; `check_parity.py` matches Hugging Face's Monad to 0.0 logit difference and cached
decoding to uncached decoding token for token. Inkling mixes through attention residuals; the gated interpolation is a
simplification.

**Bands.** B1 = layers 12–19 and B2 = 36–43 (8 layers each, low block influence and deletion KL in
`monad-weight-edit/results/layers.json`, clear of the attention-sink and loop layers 22–34), B3 = 6–21 (16 layers).

**Teacher-forced metrics** (`tf_eval.py`, `data/eval_synth.jsonl`). 384 English SYNTH rows from shard 500, which no training
run read: 128 each of memorization, math exercise and mcq. Answer NLL per token with 0%, 50% or 100% of the reference trace
kept; trace NLL; KL to the base model.

**Generation** (`gen_eval.py`, `score.py`). Greedy decoding on `monad-weight-edit`'s 192-prompt dev fixture, 1,024 new
tokens (1,536 for IFEval). With 128 scorable prompts, accuracy moved by a few prompts between arms (base 9.4%, control
13.3%, screen arms 0–14.8%) and could not separate them. Scored on MMLU and GSM8K only: IFEval's checker needs NLTK data, whose download was refused in
this environment.

**Training** (`prep_data.py`, `train.py`). SYNTH shards 1–5, English rows, 607k rows and 614M tokens; each row is the chat
prompt then `reasoning </think> answer`, loss on the target. Continued from base Monad: AdamW, learning rate 1e-4 cosine with
2% warmup, gate learning rate 1e-3, 24,576-token length-bucketed batches, bf16 autocast, torch.compile per layer, 300M tokens
(12,518 steps). Every step draws its pass count from {1, 2, 3} in a 1:2:1 ratio; the control draws the same sequence and
ignores it, so both see identical batches in identical order. Two Colab A100-40GB VMs; the control trained at 142,677 tokens/s <!-- @evidence[[results/summary.json#throughput.a100_control_tok_s]] -->, the looped arm at 129,164.

**Arms.** control (no loop); loopB1 (B1, learned gate from zero); loopB1r (B1, learned gate plus a scheduled gain rising from
0 to 1 over the first 30% of steps; stopped at step 7,300 when Oskar ended the experiment at a fixed time; scored at step 6,000, its newest checkpoint copied off the VM).

## Log

Round 1 is the baseline, Round 2 the training-free screen (3 bands × 2 or 3 passes × gate 0.1, 0.25, 0.5, 1), Round 3 the learned-gate retrofit (arm loopB1, "learned-gate arm"), Round 4 the scheduled-gate retrofit (arm loopB1r, "forced-gate arm").

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
