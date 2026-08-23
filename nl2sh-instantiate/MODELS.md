# Laptop-class generator bake-off

Same corpus, same eval, same prompts. Only the base model differs.

Issue [#52](https://github.com/oaustegard/experiments/issues/52) asks three
questions of the generator tier: which laptop-class base is best, whether the
instantiation framing beats free generation, and whether fine-tuning is worth
a training run. All three answer the same way, and the answer is size.

Every row below is the **164 leak-free rows** of the independent cyber eval,
132 distinct gold utilities, constant-utility prior **0.012**, oracle sources
(gold + 2 distractors), greedy decoding, `seed=20260819`, 64 new tokens.

## The ladder

| base | training | routing | usable | literal repro | gold-arg recall | exact |
|---|---|---|---|---|---|---|
| Gemma 3 270M | none, float32 | 0.427 | 0.427 | 0.250 | 0.173 | 0.079 |
| Gemma 3 270M | none, bfloat16 | 0.500 | 0.494 | 0.208 | 0.151 | 0.055 |
| Gemma 3 270M | **fine-tuned** | 0.610 | 0.470 | 0.469 | 0.239 | 0.030 |
| Gemma 3 1B | none, bfloat16 | 0.799 | 0.793 | 0.542 | 0.285 | 0.195 |
| **Gemma 3 4B** | none, bfloat16 | **0.957** | **0.957** | **0.917** | **0.613** | **0.402** |

*usable* = names the right utility and is not a token-repeat loop.

**A 1B instruction model with no training beats the fine-tuned 270M on every
column, and the routing column understates by how much.** Stage 1's fine-tune
bought 0.610 routing while degenerating on 22.6% of rows, so its usable output
is 0.470. The 1B degenerates on 0.6%, so nearly all of its 0.799 survives:
0.793 against 0.470 is a 1.7x gap where routing alone shows 1.3x.

The argument axis moves harder than routing. Gold-argument recall goes 0.239 to
0.613 and exact match 0.030 to 0.402 from the fine-tuned 270M to the untrained
4B. #52 guessed that "a 4B instruction model may not need [fine-tuning]". It
does not, by a wider margin than the issue expected, and 1B already clears the
bar.

## Instantiation by model size

`nl2sh-instantiate/RESULTS.md` concluded that framing the task as substitution
"does not buy routing, it buys silence". That holds at 270M and reverses at 1B:

| base | prompt | routing | bullet echo | literal repro | exact |
|---|---|---|---|---|---|
| 270M | `generate` | **0.500** | 0.000 | 0.208 | 0.055 |
| 270M | `instantiate_anchored` | 0.122 | **0.829** | 0.302 | 0.012 |
| 1B | `generate` | 0.799 | 0.000 | 0.542 | 0.195 |
| 1B | `generate_anchored` | 0.805 | 0.000 | 0.552 | 0.232 |
| 1B | **`instantiate_anchored`** | **0.848** | 0.000 | **0.688** | **0.280** |

The 270M loss was never about the framing. It answered in the shape of the
source lines it was shown — `- john — Crack password hashes: john
path/to/hashes.txt` — on 82.9% of rows, and `gold_utility()` reads the leading
`-` and scores zero. At 1B that imitation disappears entirely (bullet echo
0.000) and the framing wins every column: +0.049 routing, +0.146 literal
reproduction, +0.085 exact over free generation.

So #52's instantiation hypothesis is right and stage 2's negative result was a
270M artifact. The prompt needs a model large enough not to copy the format of
its own context.

## Execution scoring

`utility_ok` reads the leading token, which #52 opens by distrusting. Scored by
running both commands in a fixture built from the gold side — 113 files, one
directory, identical for both arms:

| base | routing | coverage | accuracy over decided | **accuracy over all** |
|---|---|---|---|---|
| 270M bfloat16 | 0.500 | 0.213 | 0.171 | **0.037** |
| 1B bfloat16 | 0.799 | 0.220 | 0.472 | **0.104** |

The 1B is **2.8x the 270M functionally**, on the same rows and the same
fixture. And the overstatement ratio itself moves: `utility_ok` inflates 13.5x
at 270M against 7.7x at 1B, so the larger model closes more of the argument gap
than the routing gap.

Coverage stays at 0.22 for both, which is the cyber corpus's ceiling under
execution rather than the fixture's: 51 golds exit 127 for offensive tooling
absent from the container by design, 34 more hit `funceq`'s deny list.

## The laptop bar

Measured with `bench.py` on 4 vCPUs, batch=1, no accelerator, each run alone on
the box (`results_bench_*_4t.json`):

| base | dtype | weights | peak RSS | TTFT | tok/s | 64 tokens | roofline | % of roof |
|---|---|---|---|---|---|---|---|---|
| 270M | float32 | 1.00 GiB | 2314 MiB | **178 ms** | 20.5 | 2.9 s | 93 | 22% |
| 270M | bfloat16 | 0.50 GiB | 1301 MiB | 446 ms | 18.6 | 3.2 s | 187 | 10% |
| 1B | bfloat16 | 1.86 GiB | 2697 MiB | 2.28 s | 4.2 | 4.3 s | 50 | 8% |
| 4B | bfloat16 | 8.01 GiB | 8210 MiB | **10.55 s** | 1.0 | 16.9 s | 12 | 8% |

#52 asks for the decode roofline first: `bandwidth / weight_bytes` at 100 GB/s.
Every arm runs at 8-22% of it, so the bottleneck is CPU matmul with no
quantised kernel rather than memory bandwidth. Real laptop hardware should do
better than every number in this table.

**Time to first token decides the product.** #52 set the bar at 12 seconds for a
whole request. The 4B spends 10.55 of that before emitting a character and 16.9
seconds to finish 64 tokens, so its 0.957 routing is not reachable inside the
bar on this class of machine. The 1B answers in 2.28 s to first token at 0.848
routing under the instantiation prompt, and that is the arm that clears both.

An earlier version of this table was measured with each model pinned to 2 of
the 4 cores while another model decoded beside it, and reported as "4 CPU
cores". Those numbers were roughly half these. `bench.py` now records
`threads`, `cpu_count`, `load1_at_start` and `other_runnable_at_start` so a row
carries its own contention evidence.

## The bfloat16 control

The 4B does not fit 15 GB at float32, so every cross-model row runs bfloat16.
The control, on a model whose float32 answer was already committed:

| 270M | routing | literal repro | exact |
|---|---|---|---|
| `generate` float32 | 0.427 | 0.250 | 0.079 |
| `generate` bfloat16 | 0.500 | 0.208 | 0.055 |
| `instantiate_anchored` float32 | 0.146 | 0.271 | 0.012 |
| `instantiate_anchored` bfloat16 | 0.122 | 0.302 | 0.012 |

bfloat16 wins by 0.073 on one condition and loses by 0.024 on the other. Greedy
decoding both times, same seed, same rows — so this is precision jitter in the
argmax path, not a systematic trade. Opposite signs across two conditions is
what rules out the "trades routing for argument fidelity" reading that either
condition alone supports.

What it establishes is a floor: **a single-run routing difference under about
0.07 is not interpretable here**, and cross-model rows have to be compared at
matched dtype. The re-run section of `RESULTS.md` independently put fine-tuning
drift at 0.012; take the larger.

## Candidates that did not produce a row

**Gemma 4 E2B** (`unsloth/gemma-4-E2B-it-GGUF`, Q4_K_M, 3.11 GB) and
**Nemotron 3 Nano 4B** (`nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`, Q4_K_M,
2.84 GB) were running when the container was reclaimed. Both are ungated and
verified fetchable — a ranged 1-byte fetch of the weight file returns 206 —
and `run_bakeoff.sh gguf` re-runs them unchanged. Google's own `q4_0` GGUF of
Gemma 4 is gated (307 to a login page); the unsloth mirror of the same weights
is not, so one 401 rules out a repo rather than a model family.

Nemotron produced three probes rather than a row, and they are the useful part:

| probe | budget | reasoning | routing |
|---|---|---|---|
| n=20 | 64 tokens | untouched | **0.000**, every row truncated mid-thought |
| n=6 | 64 tokens | `/no_think` system turn | 0.167, still 62 of 64 tokens on prose |
| n=4 | 200 tokens | untouched | 3 of 4 correct, 107 new tokens on average |

**A shared token budget is not neutral across bases.** 64 tokens is stage 1's,
every non-reasoning base meets it comfortably, and it scores a reasoning model
at zero — not because the model is bad but because the harness truncates it
mid-sentence. The documented `/no_think` control did not fire; the model reasons
in untagged prose instead.

That matters beyond this eval. At 200 tokens Nemotron needs 107 of them and
65.8 seconds to produce a command of about 15, against the 270M's 178 ms to
first token. A reasoning model is the wrong shape for this task on a laptop,
and the latency says so more decisively than any routing number would.

`run_gen_gguf.py` now strips a closed `<think>` block before parsing and
returns empty for an unterminated one, because the last line of a model's
scratchpad is not a command.

## What this does not settle

- **One family.** Every completed row is Gemma 3. The vocabulary axis #52 wanted
  — 65k Pleias, 131k Nemotron, 262k Gemma — is untested, because the two
  non-Gemma candidates did not finish.
- **Oracle sources throughout.** Retrieval surfaces the gold utility 0.555 of
  the time in the shipped stack, so every routing number here sits above what a
  deployed pipeline produces.
- **Quantisation.** Every completed row is bfloat16 or float32. The quantised
  lane exists and is validated (bullet echo and command rate match transformers
  exactly on the same rows) but has no full run behind it.
- **The eval corpus carries noise.** One request reads "Scan all ports on 192."
  against a gold command of `npm -p- 192.168.130.0`, which is not a port scan.
  That bounds what any model can score here.

## Reproduce

```bash
./run_bakeoff.sh small     # 270M dtype control, 1B, and their bench rows
./run_bakeoff.sh large     # the 4B arm
./run_bakeoff.sh gguf      # Gemma 4 E2B and Nemotron, through llama.cpp

python3 bakeoff_table.py results_*_generate.json      # cross-model, common rows
python3 funceq_ext.py --results results_1b_generate.json
```

`bakeoff_table.py` scores several runs over the intersection of their rows, so
an arm that ran 100 rows and one that ran 179 stay comparable, and prints the
intersection size beside the table.
