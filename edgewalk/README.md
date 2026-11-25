# Edgewalk: is it a sandwich?

Edgewalk is a browser game about a model that can't decide whether a food is a sandwich. You start from
a food it rates near 50/50, change the food one piece at a time, and try to keep the model undecided
while the food turns into something else. **Play it at https://austegard.com/experiments/edgewalk/web/.**
It runs in any browser; the model is 140 KB of JSON and plain JavaScript.

## Rules

- A food has nine pieces: what holds it, how it's put together, filling, sauce, temperature, what it's
  eaten with, size, where it was bought, and what the menu calls it. A move changes one piece: tap it,
  then tap a new value.
- The model's verdict decides the zone:

  | zone | P(sandwich) | the move scores |
  |---|---|---|
  | edge | 45–55% | 200 × distance |
  | fence | 35–45% or 55–65% | 100 × distance |
  | slipping | 15–35% or 65–85% | 0, and the next move must return to 35–65% or the game ends |
  | decided | below 15% or above 85% | the game ends |

- Distance is the share of the food that differs from where it started: each piece counts 0 or 1, except
  temperature and size, which count the fraction of their scale moved.
- The piece you just changed is locked for one turn. A game is 12 moves, with 3 probes that show where a
  move would land without making it.
- **Daily** gives every player the same starting food for the date; **Random** draws one. A start is one
  of twelve well-known foods (hot dog, burrito, ice cream sandwich, avocado toast…) with at most two
  pieces changed, which both Jev and the page's model put near 50%. There are 211.

## The model

The judge is a copy of [Jev](https://typesafe.ai), TypeSafe's decision model, asked the yes/no question
"This food is a sandwich." about 14,000 generated foods (`run/label.py`, no failed calls). A third of
the foods are random combinations of pieces; two thirds are 1–8 random moves away from one of the twelve
well-known foods, which is how a game moves. Some of Jev's verdicts: a deli ham and cheese 0.98, a hot
dog 0.57, an ice cream sandwich 0.67, a taco 0.18, a sushi hand roll 0.07.

`run/train.py` fits a network with two tanh layers of 32 units (3,489 parameters) to Jev's
probabilities, with 15% of the foods held out. Width, weight decay and epochs came from a sweep on the
first 6,000 labels (`results/sweep.json`).

| | held-out foods |
|---|---|
| mean \|page model − Jev\| | 0.027 |
| correlation with Jev | 0.983 |
| same zone as Jev | 78.5% |
| Jev against itself, same food asked twice (300 foods) | 0.012, same zone 92.3% |

The page model is a smoothed Jev, not Jev: on one food in five it puts you in a different zone than Jev
would. The 6,000 → 14,000 labels took held-out error from 0.032 to 0.027, so more labels would still help.

## Difficulty

On 200 held-out foods, a single move changes the page model's verdict by 4.6 points at the median; 23%
of moves change it by more than 10 points and 7% by more than 20 (`results/train.json`). Over 2,000
simulated games (`results/random_player.json`):

| player | game ends before move 12 | median score |
|---|---|---|
| random moves | 76% | 119 (median 6 moves) |
| a bot that sees every move's outcome | never | 1,572 |

A player who reads the model sits between the two.

## Files

| file | what |
|---|---|
| `run/sandwich.py` | the pieces, their values, the twelve well-known foods, the case sampler |
| `run/label.py` | asks Jev about each food; resumable; needs a Jev key (`TYPESAFE_API_KEY`) and claude-workspace's `scripts/model_advisor.py` |
| `run/train.py` | trains the page model; writes `web/sandwich.json` and `results/train.json` |
| `run/repeat.py` | asks Jev about 300 held-out foods a second time (`results/jev_repeat.json`) |
| `data/labels.jsonl` | the 14,000 foods and Jev's P(sandwich) |
| `web/` | the game: `index.html`, `edgewalk.js`, `sandwich.json` |

Run the scripts from the repo root. The page loads `gpu-bridge/web/relay.js`, which sends a game log to
a private repo only from a browser that holds that repo's token and has `?log=1` switched on.
