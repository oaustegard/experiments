# RESULTS: a Haiku swarm against a Sonnet loop at equal spend

A proof of concept after SwarmEvolve (Jiang and Karniadakis, Brown University, arXiv 2611.08483). SwarmEvolve
improves a program against an automatic scorer: each round, dozens of small agents run inside one open-weights
model, each "thinking" for a few dozen steps in the model's internal activations without writing text, and one
summarizer turns their combined state into a new program. It beats the standard evolutionary code-search
frameworks (OpenEvolve, ShinkaEvolve: loops that ask a model to rewrite the best programs so far, score the
result, and keep an archive of scored programs) given the same hour of GPU time. With Claude behind an API the
internal activations are out of reach, so this experiment tests the nearest thing possible from outside: many
cheap Haiku 5.5 calls, each proposing a short idea in text from a different angle, and one Haiku call per group
("swarm") that writes the program from the ideas. The comparison is stock OpenEvolve driving Sonnet 5.5, a
model about 20 times dearer per token, at the same dollar cost. Problems are three geometry optimisations with
a scorer and a published reference value, so a score of 1.0 means matching the reference. Effort is the
thinking-budget setting of a model call (low, medium, high). Tracking issue: oaustegard/experiments#111.

Arm names used below: **Al / Am / A** are the Sonnet loop at effort low / medium / high; **B** is the Haiku
swarm; **L** is the swarm with every call at effort low; **C** is the swarm with Sonnet as the writer; **T** is
the swarm with other group sizes. Method has the details.

## Answer

At the same dollars spent, does a Haiku swarm find better programs than a Sonnet loop? It gets near the
reference much more cheaply, and it does not end higher. Against the best Sonnet effort level for each problem,
the swarm reached 99% of the reference 17 to 33 times cheaper on circle packing ($0.013 to $0.019 <!-- @evidence[[results/headline.json#/circle/B/usd_to_0.99/max]] -->,
Sonnet at medium $0.32 to $0.44 <!-- @evidence[[results/headline.json#/circle/Am/usd_to_0.99/max]] -->) and 9 to 17 times cheaper on hexagon packing
($0.035 to $0.060, Sonnet at high $0.53 to $0.58 <!-- @evidence[[results/headline.json#/hexagon/A/usd_to_0.99/max]] -->). On the Heilbronn triangle the ranges
overlap ($0.09 to $0.50 against $0.77 to $1.02 <!-- @evidence[[results/headline.json#/heilbronn/Am/usd_to_0.99/max]] -->). Matched seed against seed, the swarm scored
higher than the Sonnet loop at effort low on every seed from $0.05 to $0.25 on all three problems, which is
the test fixed before the runs. Given its full $3, Sonnet reached the same final scores, slightly better ones on
hexagon, and got there faster in wall-clock time; the swarms ran out of their 60 minutes after about $1.

What would change this: problems that small independent edits cannot solve, or a Sonnet harness tuned for cost.

## Findings

1. **The swarm reaches 99% of the reference far more cheaply on two problems, and somewhat more cheaply on the
   third.** Dollars to 99%, swarm against the best Sonnet arm: circle $0.013 to $0.019 against $0.32 to $0.44
   <!-- @evidence[[results/headline.json#/circle/B/usd_to_0.99/min]] -->
   (effort medium), hexagon $0.035 to $0.060 against $0.53 to $0.58 (high) <!-- @evidence[[results/headline.json#/hexagon/B/usd_to_0.99/min]] -->,
   Heilbronn $0.09 to $0.50 against $0.77 to $1.02 (medium) <!-- @evidence[[results/headline.json#/heilbronn/B/usd_to_0.99/min]] -->.
   Against Sonnet at effort low, the baseline fixed beforehand, the gaps are wider: $0.36 to $0.37 on circle
   <!-- @evidence[[results/headline.json#/circle/Al/usd_to_0.99/min]] -->
   and $2.39 to $2.55 on hexagon with one seed never reaching 99% <!-- @evidence[[results/headline.json#/hexagon/Al/usd_to_0.99/min]] -->. (`results/headline.json`)
2. **Early on, every swarm seed beat its paired Sonnet seed.** Best score at $0.25: circle 0.998 to 1.0 against
   0.963 to 0.964 <!-- @evidence[[results/headline.json#/circle/Al/best_at_0.25/max]] -->;
   Heilbronn 0.928 to 1.0 against 0.790 to 0.835 <!-- @evidence[[results/headline.json#/heilbronn/B/best_at_0.25/min]] --> <!-- @evidence[[results/headline.json#/heilbronn/Al/best_at_0.25/min]] -->;
   hexagon 0.995 to 0.998 against 0.981 to 0.982 <!-- @evidence[[results/headline.json#/hexagon/B/best_at_0.25/min]] --> <!-- @evidence[[results/headline.json#/hexagon/Al/best_at_0.25/min]] -->.
   The seed-paired tables in `results/table.md` ("B minus Al") show the same at $0.05 and $0.10.
3. **Sonnet matches the swarm's final scores once it spends about $3, and gets there sooner in time.** The swarms
   stopped at the 60-minute limit after $0.89 to $1.39 <!-- @evidence[[results/headline.json#/circle/B/spent/max]] -->
   Sonnet spent its $3 in 7 to 42 minutes and ended at circle 0.998 to 1.0 (effort low), Heilbronn 1.0 on all
   three seeds (medium), hexagon up to 1.0008 (high). A stricter geometric re-check passes all three hexagon
   programs above 1.0, but only 1.0008 reproduces on a re-run; the other two optimise with randomness and re-ran
   lower (`results/strict_hex.json`).
4. **Sonnet's effort level matters on two problems of three.** On circle packing high, medium and low were within
   noise. On Heilbronn only medium reached 1.0 on all three seeds. On hexagon low never reached 0.999 (best 0.9825 to 0.9978) <!-- @evidence[[results/headline.json#/hexagon/Al/best/min]] --> <!-- @evidence[[results/headline.json#/hexagon/Al/best/max]] -->
   while high reached 0.99999 or more on all
   three. Effort high was dropped for circle seeds 2 and 3 after seed 1, so circle has one high-effort seed.
5. **Turning the swarm's writer down to effort low helps on circle packing and costs on the other two.** On circle
   all three low-effort seeds reached 0.999 by $0.14 to $0.21 <!-- @evidence[[results/headline.json#/circle/L/usd_to_0.999/min]] --> <!-- @evidence[[results/headline.json#/circle/L/usd_to_0.999/max]] -->;
   on hexagon none did, ending at 0.998 <!-- @evidence[[results/headline.json#/hexagon/L/best/max]] -->, and on Heilbronn one seed never reached 0.99. Three seeds each.
6. **The proposers did not clearly earn their cost on circle packing.** On one tuning seed the writer alone, with
   no proposers, reached 0.999 at $0.26 against $0.20 with eight proposers per swarm <!-- @evidence[[results/headline.json#/tuning/T-K4N0/usd_to_0.999]] --> <!-- @evidence[[results/headline.json#/tuning/T-K4N8/usd_to_0.999]] -->.
   SwarmEvolve's own ablation (its Appendix A.2) says the same of its latent agents: the settings with 0 to 75
   agents end the hour at similar scores. One seed on one problem; not tested on the harder two. Arm C, Sonnet as
   the swarm's writer, has one seed: 99% at $0.11, between B and the Sonnet loop.
7. **The command-line route favours the Sonnet loop slightly.** Every call carries about 1,370 tokens the CLI adds.
   Without them the swarm arms would have cost 12.5% less ($19.78 against $22.61) <!-- @evidence[[results/headline.json#/api_equiv_usd_48_runs/haiku_swarm]] -->
   and the Sonnet arms 3.7% less ($76.77 against $79.69) <!-- @evidence[[results/headline.json#/api_equiv_usd_48_runs/sonnet]] -->,
   because a swarm program takes 33 short calls.
8. **The stock hexagon evaluator accepts impossible packings.** A swarm program with a bug in its own side
   estimate reported an outer side of 1e-9 and scored 3.93 billion times the reference. The evaluator tests
   containment with an absolute tolerance on an unnormalised cross product, which any tiny outer hexagon passes.
   Eleven unit hexagons cover 11 times one hexagon's area, so the outer side can never be below the square root
   of 11; scores past that bound now count as 0. The affected run (hx-L-s2) spent its last 40 minutes refining the
   broken program. The other two evaluators were not audited. (`common.py`, `score_of`)

## Method

**Problems**, each OpenEvolve's own example with its initial program, evaluator and system prompt unchanged:
packing 26 circles in a unit square (score: sum of radii / 2.635983, the best known), the Heilbronn problem for
11 points in a triangle, and packing 11 unit hexagons in the smallest regular hexagon (both scored against
AlphaEvolve's published construction, as SwarmEvolve's Table 1 does).

**Arms**, three seeds each per problem:
- **Al, Am, A**: stock OpenEvolve 0.4.1 driving Sonnet 5.5 at effort low, medium and high (high is Sonnet's API
  default). Each problem's own config, with the model, endpoint and stopping rule changed; for the two AlphaEvolve
  problems `max_tokens` was raised from 4,000 to 16,000 so a reasoning model's answers are not cut off. Al is the
  baseline. A was dropped mid-experiment for circle packing seeds 2 and 3 once effort showed no effect there.
- **B**: the Haiku swarm. Each round runs 4 swarms in parallel: refine one of the 2 best programs, combine 3
  drawn from the best 10, explore 2 drawn from the whole archive, and start fresh from the initial program. A
  swarm is 8 Haiku proposers at effort low, each reading one parent program and one of SwarmEvolve's 14 focus
  prompts crossed with one of 9 thinking styles, and replying in at most 45 words without code; then one Haiku
  writer at effort medium (Haiku's default) that writes the complete program. Each swarm's program is scored as
  soon as it exists and joins the archive.
- **L**: B with the writer at effort low too.
- **C**: B with Sonnet at effort high as the writer (one seed; the others were dropped with A).
- **T**: other swarm shapes on circle packing, seed 11, $1 cap, 30 minutes: 4 swarms of 0, 2 swarms of 16,
  4 of 8 and 8 of 4.

**Budget and stopping.** Each run stops at $3 of list-price spend or 60 minutes of run time. Spend is each call's
actual token usage at list prices (Haiku 5.5 $0.10/$0.50 per million tokens in/out, Sonnet 5.5 $2/$10, cache reads
0.1x, 5-minute cache writes 1.25x). All calls run as headless `claude -p` children, because this environment has
no API key; each call carries about 1,370 tokens the CLI adds, and an "API-equivalent" cost without them is
recorded beside the billed one. Waits for the subscription's five-hour usage window are not counted as run time.

**Measures.** A run's curve is the best score among programs evaluated so far against the dollars spent when each
evaluation finished. Arms are compared at matched spend, seed against seed; "beats on all 3" needs a difference
above 1e-6 on every seed. Programs written by models, and OpenEvolve itself, run in the hub's `scripts/jail.sh`
(no network, user `nobody`); OpenEvolve reaches the metered model gateway through a Unix socket.

**Prior art.** The frameworks: OpenEvolve and ShinkaEvolve generate one full program per model call; the closest
feature is ShinkaEvolve's periodic meta-recommendations, one model writing up to five suggestions for the next
mutation prompt. Neither runs rounds of many short proposals feeding one writer. SwarmEvolve does, in hidden
states; its 40- and 256-token text-reasoning ablations are the nearest to this experiment, on Qwen3 8B.

**Cost and time.** 48 runs, $102.30 at list prices: $79.69 on the Sonnet arms and $22.61 on the swarms
<!-- @evidence[[results/headline.json#/total_list_usd_48_runs/all]] --> <!-- @evidence[[results/headline.json#/total_list_usd_48_runs/sonnet]] --> <!-- @evidence[[results/headline.json#/total_list_usd_48_runs/haiku_swarm]] -->.
Ten lanes (this session plus nine sibling sessions, one 4-vCPU container each) ran it in about six hours, paced
by the five-hour usage window rather than CPU.

## Log

### Round 1, 2026-11-09: harness and the circle-packing comparison

Asked: Arm A (OpenEvolve on Sonnet) against Arm B (Haiku swarm, K=4, N=8) and optional Arm C at a $3 cap and a
60-minute ceiling, three seeds, after checking whether OpenEvolve or ShinkaEvolve already supports parallel
short-proposal rounds (they do not; see Method). Built the metered gateway, the jailed evaluator, the
OpenAI-compatible socket bridge for OpenEvolve, and the swarm driver. Smoke runs: a swarm reached the best known
circle packing in round 2 for $0.058; Sonnet was at 0.978 after $0.32.

Two accounting fixes before the real runs counted. `claude -p` writes prompt cache at the 1-hour rate (2x input)
by default; `FORCE_PROMPT_CACHING_5M=1` switches it to the 5-minute rate (1.25x), which suits calls seconds apart,
and the runs made under the old rule were rerun. The five-hour usage window ran out with ten lanes going; the
gateway now waits for it to reset and leaves the wait out of the run clock, and the scheduler lets Haiku-only runs
start up to 93% of the window while Sonnet runs wait below 75%.

### Round 2, 2026-11-09: effort levels and harder problems

Asked: effort levels nearer the paper's small reasoning budgets, and one or two harder problems. Haiku 5.5's
thinking cannot be switched off or capped through `claude -p`: `MAX_THINKING_TOKENS=0` and
`alwaysThinkingEnabled: false` were ignored, and a small output cap fails the call while still billing it. Effort
low is the floor; at low the proposers thought for 0 tokens on 65% of calls. Added arm L (writer at low too) and
the Heilbronn and hexagon problems, where the paper's OpenEvolve baseline on Qwen scored 0.05 to 0.15 and 0.55 to
0.78. Then asked for Sonnet at lower effort too: added Am and Al. Circle packing showed no effect of Sonnet's
effort, so the remaining effort-high runs were dropped to save quota (`skip.txt`) and Al became the baseline.

### Round 3, 2026-11-09: complete grid

All 46 runs in scope finished, plus two effort-high Sonnet runs that had already started when that arm was
dropped. The hexagon results changed the reading of Sonnet's effort (Finding 5). The hexagon evaluator loophole
surfaced in hx-L-s2 (Finding 8). The three hexagon scores above 1.0 were re-checked in the jail with normalised
edge distances and separating-axis gaps at a 1e-9 tolerance (`jailside/strict_hex.py`): all pass, and hx-A-s2's
1.0008 reproduces, while hx-A-s1 and hx-B-s1 re-run lower (1.0004 and 0.9991) because their programs optimise
with randomness or a time budget. Plots: `results/curves.png`, `results/curves_heilbronn.png`,
`results/curves_hexagon.png`, `results/tuning.png`; tables: `results/table.md`.
