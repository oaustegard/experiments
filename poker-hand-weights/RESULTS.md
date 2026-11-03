# poker-hand-weights

A second computer opponent for the [poker game on austegard.com](https://austegard.com/games/poker/) (lobby: Computer →
**Hand-wired**): a small neural network whose every weight was written by hand, with no training, no data and no gradient
descent, in the spirit of [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/). It follows
[`weight-space-edit/`](../weight-space-edit/RESULTS.md), where the weights of the route planner in the DOOM-playing bot of
[`trm-doom/`](../trm-doom/RESULTS.md) were written the same way. The game's existing computer player (`ai.js`) estimates its
share of the pot by Monte Carlo simulation and comes in five styles. The question: can a network whose layers each compute
something you can name (card counts, made hands, a share of the pot, a decision rule) hold its own against that player's
strongest style? The network is
[`games/poker/net.js`](https://github.com/oaustegard/oaustegard.github.io/blob/main/games/poker/net.js) in the site repo.

## Answer

In two of ten formats. On held-out deals, against the existing bot's tight-aggressive style (`tag`), the network is level at six-handed no-limit hold'em
(+1.7 ± 10.1 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-tag.bb100]] --> bb/100, 95% interval) and slightly ahead at five-card draw
(+2.8 ± 2.1 <!-- @evidence[[results/headtohead.json#draw5-fl-6-net-vs-tag.bb100]] -->). It loses heads-up (−28.4 ± 16.9 <!-- @evidence[[results/headtohead.json#holdem-nl-2-net-vs-tag.bb100]] -->) and in the other seven games by 5 to 29 bb/100,
most in Stud Hi-Lo (−18.8 ± 3.3 <!-- @evidence[[results/headtohead.json#stud8-fl-6-net-vs-tag.bb100]] -->). It beats the first bot, v1, in 8 of 10 games but wins far less from weak players: +109.2 ± 31.9 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-random.bb100]] --> from a random player at no-limit hold'em,
where v1 takes +714.6 ± 84.9 <!-- @evidence[[results/reference.json#holdem-nl-6-v1-vs-random.bb100]] -->. It has no opponent model and reads every bet as strength.

1,677 ReLU units, 21,003 nonzero hand-set weights <!-- @evidence[[results/network.json#nonzero]] -->; hand reading exact on 20,000 of 20,000 hands <!-- @evidence[[results/checks.json#ladder7.agree]] -->.

An opponent model (per-seat bet and fold counts, as `tag` keeps) would most likely change it.

## Findings

1. **Level with `tag` at its home game, behind it elsewhere.** Six-handed no-limit hold'em +1.7 ± 10.1 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-tag.bb100]] --> and
   five-card draw +2.8 ± 2.1 <!-- @evidence[[results/headtohead.json#draw5-fl-6-net-vs-tag.bb100]] -->; Omaha pot-limit −28.7 ± 32.8 <!-- @evidence[[results/headtohead.json#omaha-pl-6-net-vs-tag.bb100]] --> is too noisy to call at 600 deals. It
   loses heads-up no-limit −28.4 ± 16.9 <!-- @evidence[[results/headtohead.json#holdem-nl-2-net-vs-tag.bb100]] -->, fixed-limit hold'em −6.0 ± 2.5 <!-- @evidence[[results/headtohead.json#holdem-fl-6-net-vs-tag.bb100]] -->, Seven-Card Stud
   −5.3 ± 2.4 <!-- @evidence[[results/headtohead.json#stud7-fl-6-net-vs-tag.bb100]] -->, Five-Card Stud −6.4 ± 2.5 <!-- @evidence[[results/headtohead.json#stud5-fl-6-net-vs-tag.bb100]] -->, Omaha Hi-Lo −8.7 ± 5.8 <!-- @evidence[[results/headtohead.json#omaha8-fl-6-net-vs-tag.bb100]] -->, Razz
   −9.2 ± 3.0 <!-- @evidence[[results/headtohead.json#razz-fl-6-net-vs-tag.bb100]] --> and Stud Hi-Lo −18.8 ± 3.3 <!-- @evidence[[results/headtohead.json#stud8-fl-6-net-vs-tag.bb100]] --> (`results/headtohead.json`).
2. **It beats v1, the first bot, in 8 of 10 games**, from +6.2 ± 4.9 <!-- @evidence[[results/headtohead.json#razz-fl-6-net-vs-v1.bb100]] --> at Razz to +171.6 ± 26.7 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-v1.bb100]] --> at
   six-handed no-limit hold'em, and is level at Omaha Hi-Lo −3.3 ± 8.7 <!-- @evidence[[results/headtohead.json#omaha8-fl-6-net-vs-v1.bb100]] --> and Stud Hi-Lo −3.4 ± 3.5 <!-- @evidence[[results/headtohead.json#stud8-fl-6-net-vs-v1.bb100]] -->.
   `tag` beats v1 by +271.1 ± 36.4 <!-- @evidence[[results/reference.json#holdem-nl-6-tag-vs-v1.bb100]] --> at no-limit hold'em, against the network's +171.6 ± 26.7 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-v1.bb100]] -->
   (`results/headtohead.json`, `results/reference.json`).
3. **It wins far less from weak players than either Monte Carlo bot.** At six-handed no-limit hold'em it takes
   +725.0 ± 59.8 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-caller.bb100]] --> from an always-caller, where `tag` takes +1466.3 ± 89.3 <!-- @evidence[[results/reference.json#holdem-nl-6-tag-vs-caller.bb100]] --> and v1
   +1286.9 ± 73.1 <!-- @evidence[[results/reference.json#holdem-nl-6-v1-vs-caller.bb100]] -->, and +109.2 ± 31.9 <!-- @evidence[[results/headtohead.json#holdem-nl-6-net-vs-random.bb100]] --> from a random player, where v1 takes
   +714.6 ± 84.9 <!-- @evidence[[results/reference.json#holdem-nl-6-v1-vs-random.bb100]] -->. Its value bets need a share above 0.56 heads-up, a bar set for opponents who fold and raise
   like reasonable players (`results/headtohead.json`, `results/reference.json`).
4. **A random bettor exposes the missing opponent model.** The network is level with a uniformly random player at
   fixed-limit hold'em +0.9 ± 5.1 <!-- @evidence[[results/headtohead.json#holdem-fl-6-net-vs-random.bb100]] --> and Omaha Hi-Lo −4.0 ± 4.8 <!-- @evidence[[results/headtohead.json#omaha8-fl-6-net-vs-random.bb100]] --> and loses at Stud Hi-Lo
   −9.1 ± 3.8 <!-- @evidence[[results/headtohead.json#stud8-fl-6-net-vs-random.bb100]] -->; on the same deals `tag` wins +39.3 ± 6.7 <!-- @evidence[[results/reference.json#holdem-fl-6-tag-vs-random.bb100]] -->, +56.0 ± 12.5 <!-- @evidence[[results/reference.json#omaha8-fl-6-tag-vs-random.bb100]] --> and
   +137.2 ± 8.9 <!-- @evidence[[results/reference.json#stud8-fl-6-tag-vs-random.bb100]] -->. In fixed-limit a random player bets or raises a third of the time, and the network counts
   each bet as a sign of strength (`results/reference.json`).
5. **The hand reading is exact.** The made-hand ladder gives `evalHigh`'s category on 20,000 of 20,000 random seven-card
   hands <!-- @evidence[[results/checks.json#ladder7.agree]] --> (a straight flush counted as a flush), Omaha's ladder
   matches `bestHands` on all 10,197 unpaired river boards <!-- @evidence[[results/checks.json#omaha5.unpairedAgree]] -->,
   the eight-or-better low matches on 20,000 of 20,000 <!-- @evidence[[results/checks.json#low8.agree]] -->, and the
   hold'em first-round score equals the Chen formula on 1,326 of 1,326 starting hands <!-- @evidence[[results/checks.json#chen.agree]] -->
   (`results/checks.json`).
6. **Six revisions after the first measurements, none a fitted number.** The largest gains came from fixing how a share
   against one opponent becomes a share against the table: Omaha pot-limit against `tag` went from −75 to −16 bb/100 on
   development deals when the Omaha values were rewritten against one opponent (Log, 2026-11-03).
7. **It is small and fast**: 21,003 nonzero weights and biases out of 2,399,627 dense slots <!-- @evidence[[results/network.json#params]] -->,
   and about 0.16 ms per decision in Node <!-- @evidence[[results/network.json#ms_per_decision]] --> (`results/network.json`).

## Method

- **The game** (`games/poker/rules.js` in the site repo): eight variants (Texas hold'em, Omaha, Omaha Hi-Lo, Seven-Card
  Stud, Stud Hi-Lo, Razz, Five-Card Stud, Five-Card Draw) under no-limit, pot-limit or fixed-limit betting; blinds 10/20 in
  the board and draw games, antes of 2 and a bring-in of 5 in stud. `evalHigh`, `evalLow`, `qualifies8` and `bestHands`
  are its hand evaluators.
- **The network** (`games/poker/net.js`): a ReLU multilayer perceptron with dense skip connections (each layer reads the
  inputs and every earlier layer): 774 inputs, 11 layers, 1,677 units, 21,003 nonzero weights and biases
  (`results/network.json`).
  - *Inputs*: one-hot cards (its own cards; the board; up to seven opponents' face-up stud cards; the five cards it holds at
    the draw, one slot each) and the situation as numbers. The price is the call as a share of the pot after the call.
    The aggression bucket a counts the opponents' betting: 2 for a bet this round, 2 more for a raise on top, 1 for each
    earlier bet or raise, 2 for each opponent who stood pat at the draw; 1–3 is bucket 1, 4 or more bucket 2.
  - *Writing the weights*: a `Builder` places each unit from named terms, so the line
    `b.relu('tp.K', {'pair.K': 1, 'mhas.K': 1, 'shas.A': -1}, -1)` is the weight row for "top pair of kings": the rank is
    paired, I hold one (`mhas`), no ace is on the board (`shas`). An integer threshold is two ReLUs read as their
    difference, `relu(x − k + 1) − relu(x − k)`; AND is `relu(a + b − 1)`; a value v no larger than B is switched by a 0/1
    flag g with `relu(v + B·g − B)`; a curve is a sum of ReLU ramps at fixed knots.
  - *Layers 1–6* read the hand: counts per rank and suit; thresholds on them; hand pieces (a straight in each five-rank
    window, a pair that uses its own card, top pair, Omaha's rule of exactly two hole cards and three board cards,
    eight-or-better lows, the gap terms of the Chen formula, opponents' up-card pairs); made-hand categories; a ladder of
    "at least a pair … at least quads"; and adjustments inside a category (pair rank, top pair, draws by cards to come).
  - *Layer 7* turns the hand into E, its share of the pot against one opponent still in the hand, from hand-set values
    (hold'em: nothing 0.18, a pair 0.42 to 0.71 by rank and top pair, two pair 0.80, trips 0.86, a straight 0.90), gated by
    the game being played. In the hi-lo games the high and low halves stay separate.
  - *Layers 8–9* turn E into a share against n opponents: E^k with k = n + 0.6a, nine curves for n = 1, 2, 3+ and a = 0,
    1, 2, each interpolated at eleven knots. In a split pot each half has its own curve, with about (n + 1)/2 rivals per
    half and the high hand worth 0.6 of the pot.
  - *Layer 10* is the decision rule. Call when the share beats the price (counting the price 15% lower while cards and
    chips are still to come). Bet or raise when the share clears 0.56, 0.40 or 0.30 for 1, 2 or 3+ opponents (0.12 higher
    facing a bet, 0.03 lower in position); bet the pot when it clears that by 0.17; go all in with less than 1.5 pots
    behind and a share above 0.55. Bet a strong draw about two times in three and follow a first-round raise with a bet
    heads-up about half the time, using a uniform random input as the dice. In the first round, raise first in with the top
    14% of hands from early seats up to the top 42% in the small blind, and call or re-raise a raise within narrower
    bands, from a strength percentile: the Chen formula in hold'em (a points rating for two cards) and a hand-written score
    in the other games, mapped to a percentile by running the network's own score over every starting hand
    (`tests/poker-net-pct.mjs`).
  - *Layer 11*: five action scores (fold, check or call, raise half the pot, raise the pot, all in) and a keep score for
    each card at the draw. `decode` plays the highest-scoring legal action; fixed-limit raises are the fixed amount, and
    first-round raises open to 2.5 big blinds and re-raise to about three times.
- **Checks** (the site's `tests/poker-net.mjs 20000`, `results/checks.json`): the ladder against `evalHigh` on random
  five- and seven-card hands; Omaha's ladder against `bestHands` (exact only on unpaired boards, because on a paired board
  the network leaves out the pair every player shares); the eight-or-better low against `evalLow` and `qualifies8`, and
  Omaha's low against `bestHands`; the hold'em score against the Chen formula on all 1,326 starting hands; and 1,200 whole
  hands across every game and structure, in which every action must be legal and the chips must add up.
- **Head-to-head** (`run.mjs`, which calls the site's `tests/poker-bots.mjs`): duplicate matches. Each deal is played twice
  from the same deck, the network in the even seats and then the odd ones, so card luck cancels, and every hand starts at
  100 big blinds. The result is the chips the network's seats won, divided by the hands they played, in big blinds per 100
  hands (a big blind is 20 chips), ± 1.96 standard errors over deals. Six seats (three network, three opponent) unless
  marked heads-up; 3,000 deals for no-limit hold'em, 2,000 elsewhere, 600 for the two Omaha games against `tag` and v1,
  whose simulations are slow in Omaha. Final deals start at seed offset 100,000; the weights were written and revised on
  offsets 0–299.
- **Opponents**: `tag`, the tight-aggressive style of `ai.js` (Monte Carlo share of the pot weighted by what each opponent
  has done this hand, a table model of how each seat has played in earlier hands, and bets of several sizes valued with the
  chance that everyone folds; the strongest style, and the one the game's hint line quotes); v1, the first version of
  that player (Monte Carlo against random hands, fixed thresholds), kept frozen as a baseline; `caller`, which always
  checks or calls; and `random`, which picks a legal action at random and raises the minimum, the pot or all in.
- **Reference matches** (`run.mjs --ref`, `results/reference.json`): `tag` and v1 against `caller`, `random` and each other
  on the same deals, for scale.
- **Reproduce**: with the site repo checked out beside this one (or its path in `--site` or `$POKER_SITE`),
  `node poker-hand-weights/run.mjs` then `node poker-hand-weights/run.mjs --ref`, about 25 minutes on four cores, and
  `python3 poker-hand-weights/summarize.py` for the tables.

## Log

### 2026-11-03 — build and held-out run

Asked (a sibling session started from Oskar's request): an alternate poker bot whose policy is a small network with
hand-written weights, selectable beside the existing AI and measured head to head.

Built `net.js` as described in Method. The first evaluator check found two wiring bugs before any match was played: the
full-house gate `relu(trips + two-pairs + Σ my pairs − 2)` fired on plain two pair (Σ my pairs = 2), and in Omaha two sets,
or four hole ranks each pairing the board, read as a full house, which the two-plus-three rule does not allow. Both
became weighted ANDs (`relu(10·a + 10·b + c − 20)`), and every check passed after that.

First development scan (deals 0–199, 0–299 for no-limit hold'em; bb/100 ± one standard error):

| game | vs tag | vs v1 | vs caller |
|---|---|---|---|
| Hold'em no-limit | −16.3 ± 16.2 | +178.7 ± 37.9 | +651.8 ± 91.2 |
| Hold'em fixed-limit | −6.5 ± 3.5 | +12.3 ± 7.3 | +43.6 ± 8.9 |
| Omaha pot-limit | −74.6 ± 22.5 | +78.9 ± 44.6 | +647.4 ± 107.5 |
| Omaha Hi-Lo | −30.4 ± 3.6 | −39.5 ± 5.6 | +11.1 ± 8.0 |
| Seven-Card Stud | −11.0 ± 4.6 | +4.5 ± 7.9 | +77.8 ± 11.7 |
| Stud Hi-Lo | −31.1 ± 2.9 | −17.2 ± 3.5 | +15.9 ± 6.3 |
| Razz | −11.6 ± 3.4 | −12.8 ± 4.3 | +30.0 ± 6.9 |
| Five-Card Stud | −2.2 ± 3.4 | +23.2 ± 7.3 | +70.7 ± 9.8 |
| Five-Card Draw | +4.3 ± 4.2 | +30.3 ± 7.9 | +67.5 ± 8.7 |

Six revisions followed. Each changed what a unit computes, after reading the network's decisions in the weak games (a
trace of named activations per decision, and its fold and bet rates beside `tag`'s), and each was measured on development
deals. No number was searched for or fitted.

1. Hi-lo shares: the high and low halves went through the multiway curve blended, so a nut low's share was raised to the
   power of the opponent count and it folded multiway. Each half now has its own curve. Omaha Hi-Lo vs `tag` −30 → −27,
   Stud Hi-Lo vs `caller` +16 → +33.
2. Omaha's values had been written as a share against the whole field, and the curve then discounted for the field again
   (a top set read as 9% of a four-way pot); they were rewritten against one opponent. The raise scores' tie-break bias
   (−0.25 at slope 6) had moved every value-bet threshold up by 0.04; the slopes went to 25. Omaha Hi-Lo −27 → −12, Omaha
   pot-limit −75 → −16.
3. Split pots: about (n + 1)/2 opponents contest each half, and the high hand is worth 0.6 of the pot for the times nobody
   makes a low and it scoops. Stud Hi-Lo vs `caller` +35 → +64; vs `tag` no change.
4. Aggression had multiplied every opponent's exponent, as if one bet made every hand at the table strong; it now adds to
   the exponent once (k = n + 0.6a, was n(1 + 0.6a)). Little change.
5. A single bet after a first-round raise (a continuation bet) had counted as the top aggression bucket, and the network
   folded small pairs heads-up at a fifth of the pot; it is now bucket 1. Seven-Card Stud vs `tag` −11.6 → −7.5, hold'em
   fixed-limit −4.9 → +0.3.
6. Razz had read the opponents' up cards only as a penalty for smooth low boards; a unit for "every live opponent shows a
   card that spoils a low" was added. Razz vs `tag` −15 → −10, vs v1 −13 → +3.

Then the weights were frozen (`net.js` sha256 `a6814575…`) and `run.mjs` played every pairing on deals from offset 100,000.
The held-out numbers sit within two standard errors of the last development numbers wherever both exist (Seven-Card Stud
vs `tag` −7.5 ± 5.3 on development deals, −5.3 ± 1.2 held out). Held-out results, bb/100 for the row's player ± the 95%
interval:

Hand-wired network, held-out deals (bb/100 for the network, ± 95% interval):

| game | vs tag | vs v1 | vs caller | vs random |
|---|---|---|---|---|
| Hold'em no-limit, 6 seats | +1.7 ± 10.1 | +171.6 ± 26.7 | +725.0 ± 59.8 | +109.2 ± 31.9 |
| Hold'em no-limit, heads-up | -28.4 ± 16.9 | +247.2 ± 46.3 | +513.2 ± 39.4 | +157.8 ± 54.3 |
| Hold'em fixed-limit | -6.0 ± 2.5 | +10.3 ± 5.2 | +50.1 ± 6.1 | +0.9 ± 5.1 |
| Omaha pot-limit | -28.7 ± 32.8 | +155.7 ± 68.8 | +846.0 ± 85.4 | +110.2 ± 36.4 |
| Omaha Hi-Lo | -8.7 ± 5.8 | -3.3 ± 8.7 | +40.2 ± 6.3 | -4.0 ± 4.8 |
| Seven-Card Stud | -5.3 ± 2.4 | +10.4 ± 5.3 | +81.1 ± 7.8 | +9.8 ± 4.9 |
| Stud Hi-Lo | -18.8 ± 3.3 | -3.4 ± 3.5 | +62.4 ± 6.1 | -9.1 ± 3.8 |
| Razz | -9.2 ± 3.0 | +6.2 ± 4.9 | +56.5 ± 5.4 | +37.7 ± 5.9 |
| Five-Card Stud | -6.4 ± 2.5 | +6.5 ± 4.4 | +83.4 ± 6.3 | +30.8 ± 5.4 |
| Five-Card Draw | +2.8 ± 2.1 | +40.4 ± 5.0 | +76.9 ± 5.5 | +83.1 ± 6.2 |

Reference, same deals: the tight-aggressive bot and v1 against the baselines:

tag:

| game | vs v1 | vs caller | vs random |
|---|---|---|---|
| Hold'em no-limit, 6 seats | +271.1 ± 36.4 | +1466.3 ± 89.3 | +221.6 ± 38.1 |
| Hold'em fixed-limit | +17.9 ± 7.1 | +78.4 ± 9.5 | +39.3 ± 6.7 |
| Omaha Hi-Lo | — | +54.9 ± 17.9 | +56.0 ± 12.5 |
| Seven-Card Stud | +13.9 ± 10.2 | +109.5 ± 13.8 | +52.0 ± 7.1 |
| Stud Hi-Lo | +24.8 ± 13.2 | +131.3 ± 20.0 | +137.2 ± 8.9 |

v1:

| game | vs caller | vs random |
|---|---|---|
| Hold'em no-limit, 6 seats | +1286.9 ± 73.1 | +714.6 ± 84.9 |
| Hold'em fixed-limit | +112.7 ± 7.7 | +135.2 ± 10.2 |
| Seven-Card Stud | +136.0 ± 10.9 | +159.5 ± 12.8 |
| Stud Hi-Lo | +112.5 ± 8.5 | +156.3 ± 11.7 |

After the run, the lobby option got a visible note for phone players (a tooltip alone does not show on touch screens),
and a fresh reader's pass on the write-ups added definitions for the price, the aggression bucket and E.
