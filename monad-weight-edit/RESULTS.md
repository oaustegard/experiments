# monad-weight-edit

PleIAs [Monad](https://huggingface.co/PleIAs/Monad) is a 56M-parameter reasoning model: 64 layers at width 256, an 8,192-token
vocabulary, trained on SYNTH. It answers in a chat format that opens a `<think>` block, reasons, closes it and then answers.
This experiment asks whether its weights can be edited directly, with no training, to make it better, following MIT CSAIL's
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/) post. Original and edited models were
compared on Oskar's Mac GPU through the GPU Bridge ([`gpu-bridge/`](../gpu-bridge/), a page that runs WebGPU jobs pushed from a session). It follows [`weight-space-edit/`](../weight-space-edit/RESULTS.md),
where LFM2.5-350M had no loops for a loop-removing edit to remove.

## Answer

Yes. Decoding greedily, Monad falls into a repetition loop (the same tokens repeated until the length limit) on 34.2% of a fixed 500-prompt test set <!-- @evidence[[results/summary.json#test_gpu.base.all.loop]] -->
and on 76% of the IFEval instruction-following prompts <!-- @evidence[[results/summary.json#test_gpu.base.ifeval.loop]] -->.
A looping answer never closes its think block, so it gives no answer; the MIT post removed the same failure from Qwen3.5-0.8B by deleting one direction. The edit here removes one vector direction from the outputs of the attention blocks in layers 27–34, 524,288 of
56.7M weights. On the Mac GPU it cuts loops to 5.8% <!-- @evidence[[results/summary.json#test_gpu.A8p.all.loop]] --> and raises
IFEval from 5.3% to 13.3% <!-- @evidence[[results/summary.json#test_gpu.A8p.ifeval.acc]] --> (paired p = 0.017).
MMLU (25% to 27%, chance is 25%) and GSM8K (6% to 8%) move within noise. Decode speed stays 11.8 ms per
token <!-- @evidence[[results/summary.json#bench.mwe-bench-A8p-1.B1_decode_ms]] -->.

The direction that separates looping states mostly points along the "attention sink",
an enormous vector at the first token that attention depends on; removing it everywhere gave loops on 98% of answers. The working edit projects the sink component out of the loop
direction and removes what remains from the attention output projections of layers 27–34.

What would change it: sampled decoding, or a task family the edit harms.

## Findings

1. **The edit removes the loops and helps instruction following.** Pinned test set, Mac GPU, greedy: loops 34.2% → 5.8%,
   unclosed think blocks 34.2% → 5.2% <!-- @evidence[[results/summary.json#test_gpu.A8p.all.open]] -->, paired McNemar on loops
   p = 1.5e-36. IFEval prompt-level strict 5.3% → 13.3%, loose (IFEval's lenient variant) 6.0% → 14.7% <!-- @evidence[[results/summary.json#test_gpu.A8p.ifeval.loose]] -->. MMLU 25.0% → 27.0% (p = 0.71), GSM8K 6.0% → 8.0%
   (p = 0.58) (`results/stats_test_gpu.json`). As in the MIT post, it lets the model finish answers rather than making it smarter.
2. **Monad loops a lot.** Greedy with 1,024 new tokens (1,536 for IFEval): MMLU 18.5%, GSM8K 13.3%, IFEval 76% of answers end
   in a terminal repetition; every loop also leaves `<think>` open. Answers that end on their own do close the think block
   (`results/summary.json` `test_gpu.base`). "Dev" below is a separate 192-prompt set used for every design choice.
3. **One residual direction separates looping states almost perfectly in the middle and upper layers.** Difference of means, loop positions vs
   finished answers: AUROC 0.99 at layer 33 <!-- @evidence[[results/summary.json#loopdir.auroc_loop.33]] -->, rising from 0.72
   at layer 12 and 0.96 at layer 28; at layer 33 the class means differ by 149 against a typical residual norm of 314 (`results/loopdir.json`). It predicts a loop 64 tokens before it
   starts only weakly (AUROC 0.70–0.72).
4. **That direction is mostly the attention sink, and removing it kills the model.** Cosine with token 0's residual direction
   −0.77; token 0 projects −11,790 on it at a residual norm of 15,285, against a median norm of 228. Removing it from the attention output projection (o_proj) and MLP output projection
   (down_proj) of all 64 layers gave loops on 98% of dev answers <!-- @evidence[[results/summary.json#dev_gpu.L33.all.loop]] -->; removing
   it from the five MLP neurons that write it most (22/91, 29/667, 29/668, 24/355, 29/424, the sink's massive-activation
   neurons; notation layer/neuron) gave 76% <!-- @evidence[[results/summary.json#dev_gpu.N5.all.loop]] -->, both as floods of
   newlines and quote marks.
5. **Monad does not tolerate a global projection of any direction.** Removing a random direction from the same 128 matrices
   raised dev loops from 30.2% to 51.0% <!-- @evidence[[results/summary.json#dev_gpu.L33rand.all.loop]] -->.
6. **With the sink projected out, attention writes the loop signal.** Layer 31's attention output contributes +41.8 of the
   loop-vs-normal gap along the remaining direction, layers 32 and 34 +12.8 and +10.2; MLPs contribute little
   (`results/writers_perp.json`). Dev loop rate (base 30.2%) by edit scope: layer 31 alone 28.6%, layers 31/32/34 18.2% <!-- @evidence[[results/summary.json#dev_gpu.A3p.all.loop]] -->,
   layers 27–34 7.8%
   <!-- @evidence[[results/summary.json#dev_gpu.A8p.all.loop]] -->.
7. **Side question: depth pruning is not a training-free speedup.** Decode time tracks Monad's 64 layers, so dropping layers was the other candidate edit. Removing the single least important
   layer costs 0.045 nats of next-token KL on Monad's own outputs <!-- @evidence[[results/summary.json#layers.single_kl_min]] -->,
   and 8 layers greedily chosen cost 0.57 <!-- @evidence[[results/summary.json#layers.greedy_kl.8]] --> (`results/layers.json`).
8. **The browser runtime is exact on the Mac.** onnxruntime-web 1.30 WebGPU against PyTorch CPU: max logit difference 5.5e-5,
   120 of 120 greedy tokens identical; the base dev run on the GPU produced the same text as the PyTorch CPU run for all 192 answers.
   Headless Chromium on SwiftShader gave differences up to 2.7 on the same graph, while its WASM backend matched to 3.6e-5.
9. **Speed per token is unchanged.** Batch 1 decode 11.8 ms per token edited vs 11.8–12.1 ms base, batch 8 12.7 vs 12.7–13.0 ms
   (`results/summary.json` `bench`). The edited model generated 65,811 batch steps for the test set against 73,073, since it
   stops instead of looping, and took 1,829 s against 1,853 s of Mac time. Counted as generated tokens per second, the
   evaluation ran at 218 against 254, because rows that stop early leave idle slots in their batch of 8.

## Method

- **Model**: `PleIAs/Monad` from Hugging Face, fp32 for compute. Prompt: the model card's chat template,
  `<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n`. Greedy; stop on `<|im_end|>` or EOS; 1,024 new tokens
  (1,536 for IFEval).
- **Fixture** (`fixture.py`, rows pinned in `data/fixture.json`): test = MMLU 200 (seeded sample of the test split), GSM8K test
  rows 0–149, IFEval 150 (seeded sample of 541); dev = 64 / 64 / 64 disjoint rows, used for every design choice.
- **Scoring** (`summarize.py`): MMLU, the first standalone A–D after `</think>`; GSM8K, the number after the last "answer" in the
  post-think text; IFEval, Google's checker as ported in lm-eval, on the post-think text. An answer with an open think block
  scores wrong. Loop: the answer hit the token limit and its tail is one block of 1–64 tokens repeated three or more times.
- **Loop direction** (`loopdir.py`): residual stream after each layer, teacher-forced on dev answers; loop positions two periods
  past the loop's onset vs positions in answers that finished; difference of means.
- **Writers** (`writers.py`): per sublayer and per MLP neuron, the mean projection of what it writes on the direction, loop
  minus normal. The sink direction is token 0's residual after layer 33, the layer where the loop direction separates best; the edited direction is the loop direction with that
  component removed and renormalized.
- **Edit** (`ablate.py bake --layer 33 --perp --oproj 27,28,29,30,31,32,33,34 --out models/A8p`, then `export.py models/A8p export/A8p`): `W_o ← (I − u uᵀ) W_o`, u the edited direction, for the attention output projection of layers
  27–34, written back in bf16 like the original weights. Nothing else changes.
- **Runtime** (`export.py`, `bridge/job.js`): onnxruntime-genai's builder (fp32, WebGPU target, fused GroupQueryAttention), a
  graph edit so only the needed logits are computed, weights shipped as bf16 and rebuilt to fp32 in the job with a SHA-256
  check. Batches of 8 right-padded prompts; GroupQueryAttention reads each row's length from the mask. `ort_gen.py` is the
  same loop on CPU onnxruntime and matches PyTorch to 6.7e-5.
- **GPU**: Oskar's Mac, Chrome 156, Metal (`apple metal-3`), GPU Bridge page v9, onnxruntime-web 1.30.0 from jsDelivr. Each
  test run took five jobs of about six minutes. Latency: a 37-token prompt, 128 forced decode steps, three runs, the median run.
- **CPU baseline**: PyTorch, `run_eval.py`, same fixture, HF static KV cache. Dev: all 192 answers identical in text to the
  GPU run. Test: at the first commit, 2026-11-03, 264 of 500 had run; its 200 MMLU answers score exactly as on the GPU (25.0%, 18.5% loops).

## Log

### 2026-11-03 — baseline, failed edits, the attention edit

Oskar asked, from a parent session, whether Monad could be improved by editing its weights and tested against the original
on the GPU Bridge. The CPU baseline was slow (about 220 ms per batch-32 step) and two 4-thread jobs at once thrashed the 4
vCPUs, so I moved everything that generates text to the Mac. The base test run showed loops on 34% of answers, so loops became
the target. The loop direction separated loops cleanly, but removing it everywhere broke the model, as did removing it from
the five strongest MLP writers; a random-direction control showed Monad degrades under any global projection. Those writers
were the sink's massive-activation neurons, which is how the sink component was found. Removing that component and editing
only attention outputs worked, and widening the edit from one layer to three to eight layers took dev loops from 30% to 29%,
18% and 8%. The eight-layer edit went to the pinned test set; the three-layer test run was withdrawn before it started.

| dev, Mac GPU | loops | MMLU | GSM8K | IFEval |
|---|---|---|---|---|
| base | 30.2% | 15.6% | 3.1% | 7.8% |
| all writers, loop direction | 98.4% | 0% | 0% | 0% |
| all writers, random direction | 51.0% | 9.4% | 0% | 1.6% |
| 5 MLP neurons, loop direction | 75.5% | 0% | 0% | 0% |
| attention 31, sink removed | 28.6% | 18.8% | 3.1% | 4.7% |
| attention 31/32/34, sink removed | 18.2% | 21.9% | 1.6% | 14.1% |
| attention 27–34, sink removed | 7.8% | 26.6% | 3.1% | 10.9% |

Two more variants were withdrawn before they ran, once the first rows had failed: the loop direction read at layer 40 on all writers, and the five neurons plus layer 31's attention.
