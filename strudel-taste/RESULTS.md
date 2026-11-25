# Strudel FM with taste

[Strudel FM](https://austegard.com/fun-and-games/strudel-fm.html) is a generative radio station: every four bars a seeded program writes the next phrase as Strudel code. This fork writes eight candidate phrases at each boundary and lets a model trained in the tab pick one, learning from your ♥, your skips and the phrases you let finish.

The question was whether the [browser decider](../browser-decider/RESULTS.md) could supply that model's input. The browser decider is a 10 MB decision model: Ettin-17m, a small BERT-style text encoder, fine-tuned to answer typed questions about a piece of text, running on WebGPU in the page. Here its weights are kept frozen and it is used only as an encoder: it reads a description of each candidate, and the 256-number hidden state at the candidate's `[MASK]` token becomes the candidate's vector. The alternative input is 31 numbers counted from each candidate's code.

Page: https://austegard.com/experiments/strudel-taste/web/ (counted features; `?brain=ettin` for the Ettin vectors).

## Answer

Can frozen browser-decider vectors teach a radio station a listener's taste from a few dozen reactions? Not well: 31 counted features do it far better. The test used simulated listeners, not people. Each heard 60 phrases, and every pick is scored by where the played candidate ranks among its eight in that listener's taste: 0.5 is a random pick, 1.0 would be always the favourite (ties put the reachable ceiling lower). Scores are for decisions 41–60, after 40 reactions; all-60 averages (Log) rank the arms the same.

For listeners whose taste weights the counted properties, the feature model scored 0.81 <!-- @evidence[[results/sim.json#test.linear.feats.t41_60]] -->, the Ettin vectors 0.56 <!-- @evidence[[results/sim.json#test.linear.desc.t41_60]] --> and random picks 0.51 <!-- @evidence[[results/sim.json#test.linear.random.t41_60]] -->. For listeners whose taste no counted feature can see (chord roots), Ettin scored 0.55 <!-- @evidence[[results/sim.json#test.roots.desc.t41_60]] -->, a little above random and far below the 0.73 ceiling <!-- @evidence[[results/sim.json#ceiling.roots]] -->. The page picks with the counted features; Ettin is opt-in.

The station also limits any model: the eight candidates of a set differ less than phrases across a session (median within-set standard deviation 0.33 of the overall one <!-- @evidence[[results/summary.json#within_over_total_std_median]] -->). Real listeners, wider candidates or a fine-tuned encoder could change the answer; none was tried.

## Findings

1. **Counted features learn fast.** Feature-model percentile over decisions 41–60: 0.81 for linear listeners <!-- @evidence[[results/sim.json#test.linear.feats.t41_60]] --> (ceiling 0.995 <!-- @evidence[[results/sim.json#ceiling.linear]] -->), 0.73 for categorical ones <!-- @evidence[[results/sim.json#test.categorical.feats.t41_60]] --> (ceiling 0.88 <!-- @evidence[[results/sim.json#ceiling.categorical]] -->). Random stays near 0.51. (`results/sim.json`, test listeners)
2. **Ettin vectors stay near random.** Description vectors: 0.56 linear <!-- @evidence[[results/sim.json#test.linear.desc.t41_60]] -->, 0.60 categorical <!-- @evidence[[results/sim.json#test.categorical.desc.t41_60]] -->. Code vectors (0.55–0.58) and the description vectors projected on 32 principal components (0.56–0.57) did no better. (`results/sim.json`)
3. **Text-only taste is where Ettin edges ahead, slightly.** Chord-root listeners: description vectors 0.55 <!-- @evidence[[results/sim.json#test.roots.desc.t41_60]] -->, counted features 0.49 <!-- @evidence[[results/sim.json#test.roots.feats.t41_60]] --> (they cannot see roots), random 0.51. The margin is a few standard errors and a fraction of the 0.73 ceiling. (`results/sim.json`)
4. **Nobody learns the interaction listener.** Busy drums only under a sparse lead: every arm 0.53–0.55 <!-- @evidence[[results/sim.json#test.interaction.desc.t41_60]] -->, ceiling 0.80. Both models are linear. (`results/sim.json`)
5. **The candidate sets are narrow.** Per counted feature, the standard deviation within a set is 0.09–0.89 of its standard deviation over all candidates, median 0.33 <!-- @evidence[[results/summary.json#within_over_total_std_median]] -->; 135 of 400 sets contain at least two candidates with identical descriptions. (`results/summary.json`, harvest)
6. **The CPU reference matches the page's vectors.** Maximum relative difference 3.67e-07 <!-- @evidence[[results/summary.json#cpu_vs_webgpu_max_rel_diff]] --> over 2 sets and both texts. SwiftShader, the software WebGPU of headless Chromium in this container, needed 134 s per set for both texts <!-- @evidence[[results/summary.json#swiftshader_s_per_point_both_texts]] -->, which is why the simulation used the CPU reference. Real-GPU latency for this page was not measured; the browser decider measured 19 ms for a case of similar size. (`run/embed.py --check`)

## Method

**The station.** [Strudel FM](https://austegard.com/fun-and-games/strudel-fm.html) writes each four-bar phrase as Strudel code from a seeded random generator. The fork in `web/index.html` snapshots the generator state (seed, phrase history, chord voicing, melody memory, key, cues) and writes eight candidates from it. Candidate 0 is the phrase plain Strudel FM would have written. The other seven use another seed, and some also nudge the energy one step (40%) or cue a breakdown, a drop or a key change (12%). The chosen candidate is installed together with the generator state it produced, so the station carries on from it. The fork writes 2 bars ahead (Strudel FM writes 6), so a reaction in a phrase's first two bars can steer the next one. Skip cuts at the next bar and picks the replacement with the model as updated by that skip.

**Two inputs.** The page's default input is 31 numbers counted from each candidate (19 numeric: energy, energy tilt, hits per bar for kick, snare, closed and open hats and percussion, bass, chord and lead counts, lead step share, height and range, texture, chord changes, chord qualities; then 12 one-hot flags for section and mode). The alternative, `?brain=ettin`, is an Ettin vector: each candidate becomes a sentence (section, energy, key, chord symbols, the same counts in words, lead register, texture) or, with `?text=code`, its first bar of Strudel code, and the browser decider encodes it as a one-option choice question, `[CLS] choice: <instruction> [SEP] [MASK] candidate: <text> [SEP] <station and tempo> [SEP]`. The 256-number final hidden state at the `[MASK]` is the vector; the decider's own answer layer is not used.

**Learning.** Over either input, a weighted logistic regression with an L2 penalty λ, trained by full-batch gradient descent from the previous weights after every reaction. ♥ is y=1 with weight 1, Skip is y=0 with weight 1, and a phrase left to finish is y=1 with weight 0.3. The next phrase is sampled from softmax(score / τ), where τ is a temperature, mixed with 5% uniform. With no labels every score is 0, so the station starts by picking evenly. In `?brain=ettin` mode the counted-feature model also trains, and chooses whenever a set's GPU vectors are not ready or WebGPU is missing.

**Simulation.** `run/harvest.mjs` drives the page headless and records 400 decision points: six stations (Sleep left out), a new one every ten points, random mood and energy, and a random walk through the candidates. `run/embed.py` computes the same vectors on CPU with the browser decider's NumPy reference (`browser-decider/run/pack.py`, `Ref`). `run/sim.py` then runs simulated listeners: each has a hidden utility, is shown T = 60 random sets in turn, and reacts to the played phrase with like above its 75th utility percentile, skip below its 35th, otherwise lets it play; 10% of reactions are random. Four listener families:

| family | utility | which input it favours |
|---|---|---|
| linear | random weights on the 19 numeric counted features | the counted features (its own model class) |
| categorical | +1 for each of two liked modes and two liked sections, +0.5 for one energy nudge direction | both; the features carry these as one-hots |
| roots | number of chords whose root is one of three liked pitch classes | the Ettin inputs: no counted feature carries chord roots |
| interaction | busy drums are good under a sparse lead and bad under a busy one | neither: both models are linear |

The simulation compares five arms, each a way of picking among the eight: random; `feats`; `desc` (Ettin vector of the sentence); `code` (Ettin vector of the first bar's code); `desc32` (the `desc` vector on its top 32 principal components, computed over the pool). Every arm's inputs are standardised over the pool. τ ∈ {0.25, 0.5, 1} and λ ∈ {0.1, 1, 10} are chosen per arm on 40 listeners per family (seed 1); the tables report 40 other listeners per family (seed 2). τ = 0.25 and λ = 0.1 were chosen at the low edge of their grids for the feature and description arms, so slightly better settings may lie below them.

**Metric.** The percentile of the played candidate's utility within its own set of eight: 0.5 is a random pick, and the ceiling is below 1 where the favourite ties with another candidate. Reported over all 60 decisions and over decisions 41–60, with the standard error across listeners.

## Log

### 2026-11-25 — build, harvest, simulate

Oskar picked the "generator with taste" idea from a brainstorm on putting an in-page decision model into something interactive. Built the fork of `oaustegard.github.io` `fun-and-games/strudel-fm.html` (Strudel FM's generator unchanged apart from writing into a scratch map, a URL guard and the taste hooks). Smoke test (`run/smoke.mjs`, both brains): 8 candidates shown, ♥ and Skip each stored a label, Skip cut from bar 8 to a picked replacement at bar 9, no console errors.

Harvest: 400 points in 0.1 s each with no GPU work. WebGPU vectors on SwiftShader took 134 s per point, so `run/embed.py` computed them on CPU (about 3 s per point, 22 minutes in all) after a parity check on two points.

The first simulation run crashed with NaN: gradient descent at a fixed step of 0.5 diverges on 256 standardised dimensions, whose curvature bound is near 64. The page had the same bug. Both now take a 1/L step from the bound. Chosen on dev listeners: τ = 0.25 for all four arms, λ = 0.1 for feats and desc, 1 for code, 10 for desc32.

Test listeners, percentile of the played candidate, all 60 decisions / decisions 41–60 (standard error over 40 listeners, all decisions):

| listeners | ceiling | random | feats | desc | code | desc32 |
|---|---|---|---|---|---|---|
| linear | 0.995 | 0.509 / 0.510 | **0.745 / 0.810** (±0.010) | 0.561 / 0.560 | 0.566 / 0.583 | 0.543 / 0.565 |
| categorical | 0.876 | 0.507 / 0.509 | **0.701 / 0.735** (±0.016) | 0.569 / 0.595 | 0.540 / 0.550 | 0.552 / 0.575 |
| roots | 0.732 | 0.508 / 0.505 | 0.494 / 0.489 | **0.541 / 0.548** (±0.006) | 0.511 / 0.525 | 0.521 / 0.539 |
| interaction | 0.795 | 0.503 / 0.500 | 0.535 / 0.535 | 0.539 / 0.546 | 0.527 / 0.527 | 0.525 / 0.531 |

Reading at the time, untested: the decision model's encoder was fine-tuned for business-workflow decisions, and its [MASK] state may carry little about counts written as text ("kick 4, closed hats 3.8"); with 60 labels and 256 dimensions a linear head cannot pull them out. The page switched to the counted features by default (`?brain=ettin` keeps the comparison), with τ and λ from the dev choice.
