# nl2sh-instantiate: does a 270M model do better at substituting than at generating?

Issue [#52](https://github.com/oaustegard/experiments/issues/52) argues the
on-device shell helper's remaining loss is not free generation but
**instantiation** — the user's literals substituted into a documented example —
and that a small model should be able to do the narrower task. §6 of
[`../nl2sh-dense/RESULTS.md`](../nl2sh-dense/RESULTS.md) is the evidence it
rests on: under oracle sources an exemplar is worth **+0.189** routing while the
*choice* of exemplar is worth **zero**, which reads as a model copying a template
rather than composing one.

This directory tests the reading on Gemma 3 270M, the stage-1 generator, holding
the model, the sources, the distractors, the decode and the seed fixed and
varying only the instruction.

<!-- FINDINGS -->

**Headline: the instantiation framing does not buy routing, it buys silence.**
Zero-shot the substitution prompt *loses* badly — 0.146 against 0.500 — and the
loss is a format artifact: the model answers in the shape of the source lines it
was shown, bullet included, on 0.774 of rows. One epoch of stage-1 fine-tuning
erases that entirely (0.774 to 0.000), and past it the two prompts route the
same: paired on identical rows, 23 wins to 20, p = 0.76.

What does separate is garbage. Token-repeat loops fall **0.183 to 0.049**, a
3.7x reduction, and *usable* — routes correctly and is not a loop — goes 31 wins
to 16, p = 0.040, +0.092. `nl2sh-selfhist/MODELS.md` named degeneracy as "the
real ceiling to chip at" after `repetition_penalty=1.3` bought a comparable
reduction at a cost of 0.118 routing. Framing the task as substitution buys the
same reduction at no routing cost.

The published NL2SH benchmark reproduces the direction on rows built to be
executed, once its 0.393 always-`find` prior is subtracted: routing 0.854 to
0.866, degeneracy 0.144 to 0.085.

So the answer to the title is **no, and the question was slightly wrong**. A
270M model is not better at substituting than at generating in the sense of
naming the right utility more often. It is better behaved when told it is
substituting, and on this task the difference between an answer and a loop is
worth more than the routing column that stage 1 was reporting.

> **Revised by the bake-off ([`MODELS.md`](MODELS.md)).** That answer holds for
> 270M and reverses at 1B, where bullet echo falls to 0.000 and
> `instantiate_anchored` beats `generate` on every column — routing 0.848
> against 0.799, literal reproduction 0.688 against 0.542. The negative result
> below was the model imitating the format of its own context, not the framing
> failing.


## What is held fixed

Every run uses **oracle sources** — the gold utility plus two distractors, one
tldr example each, shuffled — because #52 puts retrieval out of scope at 0.555
gold-in-sources against an 0.640 oracle ceiling. Greedy decoding, no repetition
penalty, 64 new tokens, `seed=20260819`, and stage 1's fenced-or-bare command
parser, all carried over from
[`../nl2sh-retrieval/gemma_arm.py`](../nl2sh-retrieval/gemma_arm.py) unchanged.

The eval is the **164 leak-free rows** of the independent cyber eval — real
commands from the Zenodo/UCI corpus, request text written by Gemini without
naming the utility, 132 distinct gold utilities, **constant-utility prior
0.012**. `nl2sh-selfhist`'s 34 rows are a subset, so stage-1 numbers quoted here
are on that subset and are marked as such.

`generate` is stage 1's prompt byte for byte. `instantiate` names the
substitution and appends the literals `extract_params.py` lifted from the
request at 0.971 precision. `instantiate_bare` keeps the framing and drops the
literals, so a gain attributes to one half or the other.

## Metrics past `utility_ok`

#52 opens by distrusting `utility_ok`, which reads the leading token: stage 1
reported 0.250 end-to-end routing under it while a read of all 41 outputs put
genuinely-runnable-and-correct nearer 0.05. Five columns sit between those two
numbers, and `score.py` reports them for every run identically:

| column | what it counts |
|---|---|
| routing | `utility_ok` — the leading token matches the gold utility |
| utility named | the gold utility appears as a token anywhere in the raw generation |
| bullet echo | the output is a source line (`- john — Crack hashes: john …`), not a command |
| degenerate | a repeated adjacent token, or a bigram three times over — routes right, does nothing |
| literal reproduction | of the values lifted from the request, how many come back verbatim |
| gold-arg recall | of the gold command's tokens after argv[0], how many appear in the prediction |

Scoring the copy as its own metric is `monad-bsky`'s rule, in `METHODS.md`: a
combined arguments-correct number hides which half is broken.

## Zero-shot: the instantiation prompt names the right utility and does not emit a command

All six cells, `unsloth/gemma-3-270m-it` with no training, 164 leak-free rows:

| prompt | routing | utility named | bullet echo | literal repro | gold-arg recall | exact |
|---|---|---|---|---|---|---|
| `generate` (stage 1) | 0.427 | 0.451 | 0.000 | 0.250 | 0.173 | 0.079 |
| `generate_anchored` | **0.500** | 0.506 | 0.000 | 0.156 | 0.151 | 0.085 |
| `instantiate` | 0.165 | 0.549 | 0.768 | **0.292** | 0.194 | 0.006 |
| `instantiate_anchored` | 0.146 | 0.561 | 0.774 | 0.271 | **0.201** | 0.012 |
| `instantiate_bare` | 0.122 | 0.561 | 0.848 | 0.271 | 0.169 | 0.000 |
| `instantiate_anchored_bare` | 0.140 | **0.579** | 0.829 | 0.240 | 0.173 | 0.012 |

Read the first column alone and instantiation loses by 0.35. Read the second and
third together and it wins the part of the task it was meant to win: the
instantiation prompts find the gold utility **0.549–0.579** of the time against
free generation's 0.451–0.506, and reproduce the request's own literals
**0.240–0.292** against 0.156–0.250 — while **77–85% of their outputs are source
lines rather than commands**. `- john — Crack password hashes: john
path/to/hashes.txt` for *"crack the password hashes in crack.txt"*: the right
utility, in the shape of the block the model had just been shown.
`gold_utility()` reads the leading `-` and scores it zero, which is the correct
behaviour for a metric that asks whether the first token is a command.

**The `Command:` anchor was the control and it failed at its job.** Appending a
cue line was meant to disambiguate the output slot and separate format imitation
from framing; bullet echo moved 0.768 to 0.774. What it did instead was help
free generation, 0.427 to 0.500, which is a prompt-tuning result for stage 1's
prompt and not an answer to this question. A cue token does not override format
imitation at 270M.

Handing over the extracted literals is worth a little on both instantiation
axes — 0.292 against 0.271 literal reproduction, 0.165 against 0.122 routing —
and the bare variants echo *more* (0.829–0.848). The literal list gives the model
something to do other than restate the source.

This splits into two questions the zero-shot grid cannot answer together. Whether
the model can identify the utility and copy the operands: measured, and
instantiation is ahead on both. Whether it can be made to emit a command instead
of a documentation line: not a prompt property. Stage 1 answered that once
already — its zero-shot failure was **output shape, not capability**, and one
epoch of fine-tuning took the same base from 0.026 to 0.706 on the n=34 slice.


## Fine-tuned: routing is unchanged, the garbage rate falls by 3.7x

Stage 1's recipe, twice — 600 NL2Bash rows whose gold utility has a tldr page,
gold plus two distractors, one epoch, full-parameter fp32 AdamW at 1e-4, loss
masked to the model turn — with only the user turn swapped. 16.7 minutes for
`generate`, 20.9 for `instantiate`, on 4 CPU cores.

| n=164 leak-free | routing | usable | degenerate | literal repro | gold-arg recall | bullet echo |
|---|---|---|---|---|---|---|
| zero-shot `generate_anchored` | 0.500 | 0.500 | 0.000 | 0.156 | 0.151 | 0.000 |
| zero-shot `instantiate_anchored` | 0.146 | 0.146 | 0.000 | 0.271 | 0.201 | 0.774 |
| FT(`generate`) | 0.598 | 0.506 | 0.183 | 0.604 | **0.275** | 0.000 |
| FT(`instantiate`) | **0.616** | **0.598** | **0.049** | **0.646** | 0.222 | 0.000 |

*usable* = routes correctly **and** is not a token-repeat loop.

Paired on the same rows, **routing does not separate**: 23 wins to 20, p = 0.76.
*Usable* does — **31 wins to 16, p = 0.040**, +0.092. The whole difference is the
degeneracy collapse, **0.183 to 0.049**, and that is the number
[`../nl2sh-selfhist/MODELS.md`](../nl2sh-selfhist/MODELS.md) named as "the real
ceiling to chip at" after `repetition_penalty=1.3` bought a 3.7x reduction at a
cost of 0.118 routing. Framing the task as substitution buys a similar reduction
at no routing cost.

Two rows, same request, both models fine-tuned the same way:

| request | FT(`generate`) | FT(`instantiate`) |
|---|---|---|
| find rockyou.txt.gz on my system | `locate my -o rockyou.txt \| xargs -0 -0 -c -0` | `find / -name "rockyou.txt.gz"` |
| run a quick port scan on 172.18.1.5 | `nmap -v1 \| xargs -0 -0 -c '172.18.1.5.` | `nc \| xargs -0 grep -i 's/ /g'` |

The second is the honest half of the pair: instantiation is not right there
either, it is merely not looping.

**Training erased the bullet echo, 0.774 to 0.000.** That settles what the
zero-shot grid could not: the source-line imitation was output shape, not an
inability to hold the instruction, and one epoch fixes it — the same finding
stage 1 recorded when its zero-shot 0.026 became 0.706, and `monad-bsky`'s
0.000 to 0.481 before that.

Fine-tuning is also what installs literal copying at all: **0.250 to 0.604** for
free generation, 0.271 to 0.646 for instantiation. Against `monad-bsky`'s 56M
model at 0.51 on the same operation, a 268M model with a 262k vocabulary
reproduces two thirds of the operands it is handed. Gold-argument recall runs the
other way — 0.275 for `generate` against 0.222 — so the instantiation arm copies
what the request contains and supplies less of what the request implies, which is
what a substitution prompt should be expected to do.

### A metric that changed a number, disclosed

The first degeneracy detector counted only whitespace-token repeats, so it missed
loops inside a single token — `apt -f -o my_my_my_my…`, `user.html.html.html…`.
Adding the intra-token rule left FT(`generate`) at 0.183 and moved
FT(`instantiate`) from 0.024 to 0.049: it penalised the arm it was found on and
narrowed the gap being claimed, from 7.6x to 3.7x. The rule anchors on a letter,
because without that it fires on `100.100.100.4` and `8.8.8.8`.

### Re-run: the effect is reproducible, the exact numbers are not

The container that produced the tables above was reclaimed, so everything was
re-run from the recovered scripts. **All six zero-shot conditions came back to
the digit** — 0.427, 0.500, 0.165, 0.146, 0.140, 0.122 — because inference here
is greedy, seeded, and bit-reproducible on CPU.

Fine-tuning is not. Same seed, same 600 rows, same 300 steps, and the two runs
differ:

| n=164 leak-free | first run | re-run |
|---|---|---|
| FT(`generate`) routing | 0.598 | 0.610 |
| FT(`instantiate`) routing | 0.616 | 0.616 |
| FT(`generate`) degenerate | 0.183 | 0.226 |
| FT(`instantiate`) degenerate | 0.049 | 0.055 |
| FT(`generate`) usable | 0.506 | 0.470 |
| FT(`instantiate`) usable | 0.598 | 0.585 |
| final-step loss, `generate` | 2.1403 | 1.3017 |

The **conclusion is unchanged and slightly stronger**: routing still does not
separate (22 wins to 21, p = 1.000, against 23–20 and p = 0.76), *usable* still
does and by more (**38 wins to 19, p = 0.016, +0.115**, against 31–16, p = 0.040,
+0.092), and the degeneracy collapse is **0.226 -> 0.055**, 4.1x, against 3.7x.

Two things follow. **Quote a fine-tuned number here to two decimals, not three** —
run-to-run drift on the routing column is about 0.012, which is two rows.
And **the effect being replicated is the degeneracy gap, not any single cell**;
it survived a full re-run with different weights underneath it, which is a
stronger claim than the first run could make on its own.

Same pair of rows, this run's weights:

| request | FT(`generate`) | FT(`instantiate`) |
|---|---|---|
| show all files including hidden ones | `ls -d -d -d -d -d -d -d -d -d …` | `ls -l` |
| recover the password for invoices2019.zip | `echo …fasttrack.txt \| awk -f 's/ // // // …` | `fcrackzip -b /usr/share/wordlists/fasttrack.txt` |
| open authorized_keys in a text editor | `nano -e 's/ -e/ // // // // …` | ``nano -c `read -s`` |

The third row is again the honest one: instantiation is not right, it is merely
not looping.

## The published benchmark: same direction, and a prior that eats the headline

`westenfelder/NL2SH-ALFA` (MIT, [arXiv:2502.06858](https://arxiv.org/abs/2502.06858))
is the external set #52 names. Its test split is 300 requests, each with **two**
acceptable gold commands and paths under `/testbed` chosen so the commands
actually run — the property the cyber corpus lacks. 271 rows do not name their
utility; 270 of those have a tldr page and are scored here.

| n=270 leak-free | routing (all) | routing (non-`find`, n=164) | usable | degenerate | bullet echo |
|---|---|---|---|---|---|
| zero-shot `generate` | 0.419 | — | 0.419 | 0.000 | 0.000 |
| zero-shot `generate_anchored` | 0.459 | 0.433 | 0.456 | 0.007 | 0.000 |
| zero-shot `instantiate_anchored` | 0.096 | — | 0.096 | 0.000 | 0.681 |
| FT(`generate`) | 0.885 | 0.811 | 0.804 | 0.100 | 0.000 |
| FT(`instantiate`) | **0.907** | **0.848** | **0.859** | **0.048** | 0.000 |

Re-run figures. The three zero-shot rows are identical to the lost run to the
digit; the two fine-tuned rows moved with their weights — 0.911/0.919 routing
there against 0.885/0.907 here, degeneracy 0.144/0.085 there against 0.100/0.048
here. The gap between the arms holds its sign and its size in both.

**The constant "always-`find`" prior is 0.393** — 106 of 270 golds — so 0.911 is
mostly the skew, and the non-`find` column is the one that carries information.
`find` is 109 of the benchmark's 300 rows; NL2Bash has the same lean, which is
why `score_gate_ft.py`'s rule of printing the prior beside the headline applies
to the external set too rather than only to ours.

Read on the non-`find` slice, the benchmark says what the cyber eval said and
says it on rows built to be executed: routing separates a little (0.811 to
0.848), degeneracy halves (0.100 to 0.048), and *usable* gains +0.055. The lost
run measured 0.854 to 0.866 and 0.144 to 0.085 on its own weights. Both
fine-tuned arms sit far above the 0.459 zero-shot ceiling, and the zero-shot
instantiation arm collapses to the same bullet echo — 0.681 of rows — that the
cyber grid found.

The benchmark's own metric is execution plus a model judge, and it reports
**74% for GPT-4o**. Nothing here is comparable to that number until the
execution scoring below is read, and none of these routing figures should be
put beside it.

## Execution scoring: the first functional number, and it is 0.055

`funceq.py` was blocked in stage 1 — 17 of 40 rows INCONCLUSIVE, because the
fixture had none of the files the commands name, which is the harness measuring
its own sandbox. `funceq_ext.py` takes the first of the two routes #52 names:
build the fixture from the paths the **gold** commands mention before anything
runs. That is neutral between the arms — both sides meet the same tree, and a
prediction that invents a different filename still fails.

Zero-shot `generate`, the 164 leak-free cyber rows, 113 files and 1 directory
created:

| | |
|---|---|
| decided (EQUIVALENT + DIFFERENT) | 36 of 164 — **coverage 0.22** |
| functional accuracy over decided | 0.250 |
| **functional accuracy over all** | **0.055** |
| routing (`utility_ok`) over all | 0.427 |

**`utility_ok` overstates by about 8x**, and 0.055 lands on top of the estimate
#52 arrived at by reading all 41 stage-1 outputs by hand — "genuinely runnable
and correct is nearer 0.05". The metric the issue distrusted deserved it.

Widening the fixture further will not raise the 0.22. The 128 undecided rows
break down as: **48 golds exit 127** — `nmap`, `john`, `fcrackzip`, `msfconsole`
are absent from the container by design and installing offensive tooling to
score an eval is not a trade worth making — **34 golds and 19 predictions hit
`funceq`'s deny list** (`curl`, `ssh`, `scp`, `kill`), and 19 golds exit 1 in a
fixture that cannot hold their real state. That is the cyber corpus's ceiling
under execution, not this fixture's, and it is why NL2SH-ALFA is the corpus
where functional equivalence can actually decide.

### On comparing anything here to the paper's 74%

The benchmark reports 74% for GPT-4o, and that number comes from InterCode-ALFA:
their container image, their command set, and **a model judge for the rows
execution cannot separate**. An execution-only score with no judge is a floor
computed under a different rule. Wherever a number from this directory appears
beside it, say that.

## Caveats

- **Every number in this file was measured, and then its container was lost.**
  The session that ran the grid wedged on a foreground wait for the last
  background job (a `run_in_background` poll loop, which suspends the turn
  rather than freeing it) and was reclaimed before it committed. Scripts,
  prompts, metrics and results tables here were recovered verbatim from that
  session's transcript. **The re-run has since completed and every
  `results_*.json` is committed**, so the tables are backed by artifacts a
  reader can re-derive. All six zero-shot conditions came back to the digit;
  the fine-tuned arms did not, and the *Re-run* section above tabulates both
  and says what survived.

- **One model, one seed, one eval family.** Every number in *this file* is
  Gemma 3 270M. The bake-off #52 asks for has since run and lives in
  [`MODELS.md`](MODELS.md), and two of its findings revise this file rather
  than extending it: a zero-shot 1B beats the fine-tuned 270M on every column,
  and the instantiation framing **wins** at 1B (0.848 routing against 0.799)
  once the model is large enough not to imitate the source-line format. The
  headline below — "the instantiation framing does not buy routing" — is a
  270M result, not a general one.
- **Oracle sources throughout.** Retrieval surfaces the gold utility 0.555 of
  the time in the shipped stack, so every routing number here is above what the
  deployed pipeline would produce.
- **The cyber eval's utility names come from a corpus this container cannot
  run.** Execution decides 0.22 of its rows and no wider fixture changes that;
  read the coverage beside the functional accuracy rather than reading the
  accuracy as a capability.
- **The container is not a laptop.** 4 CPU cores, fp32 weights, no quantised
  build. The bar in `bench.py` is measured here and labelled as such.

## Reproduce

```bash
# corpora — both public, cloned anonymously
git clone --depth 1 https://github.com/tldr-pages/tldr.git
git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
curl -sL -o test.csv \
  "https://huggingface.co/datasets/westenfelder/NL2SH-ALFA/resolve/main/test.csv"
python3 alfa_prep.py --csv test.csv

# zero-shot grid on the cyber eval (~3 min per condition on 4 CPU cores)
./run_all_it.sh
python3 score.py results_it_*.json

# fine-tune under each prompt, then evaluate under the prompt it was trained on
./run_ft.sh

# the published benchmark's test split
./run_alfa.sh
python3 funceq_alfa.py --results results_alfa_*.json

# execution scoring on the cyber eval, fixture built from the gold commands
python3 funceq_ext.py --results results_it_generate.json

# the laptop bar
python3 bench.py
```
