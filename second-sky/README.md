# second-sky

**The Second Sky** is a full set of constellations for an invented people, the Ilvari, drawn on the real stars. There are 140 figures, each with a myth, a three-line verse, named stars and directions for finding it, plus an epic told as a guided tour. The whole thing is one interactive planetarium page, built by 1,843 separate Claude Haiku 5.5 calls for $5.80 of model cost.

- Live: **https://muninn.austegard.com/scratch/second-sky/**. The source is [`second-sky.html`](second-sky.html), self-contained at 1.1 MB; `python3 assemble.py --site DIR` writes the hosted version with its document head and social card.
- Drawing improvement, first draft (top) vs final (bottom): [`before-after.png`](before-after.png)

The stars are real: 5,044 stars to magnitude 6 from the d3-celestial copy of the Hipparcos-derived catalogue, with its Milky Way outline. Everything else was invented by the agents, inside a one-page world bible ([`bible.md`](bible.md)). The bible sets a pantheon of eight, five houses of the sky, an alphabet, and a register: spare, maritime, a grandmother mending a net.

## Pipeline

| stage | calls | cost | what a call does | what checks it |
|---|---|---|---|---|
| seeds | 15 | $0.06 | 30 constellation concepts per house × angle | dedup by gloss |
| draw | 140 | $1.00 | one patch of 7–23 stars → name, star-id stick figure, outline drawing, anatomy, myth, verse, star names | `phase1_draw.validate`: real ids, one connected figure, 4–16 stars, alphabet, word counts. 140/140 passed first time |
| critique ×2 | 280 + 280 | $0.76 | blind reader names an unlabelled render; judge compares the guess to the subject and gives advice | — |
| redraw ×2 | 93 + 80 | $1.26 | redraw anatomy, lines and figure with the rendered image in hand | `validate` again |
| best-of-3 | 174 + 174 + 174 | $1.61 | 3 fresh redraws for each of the 58 figures still unrecognised, each blind-read and judged | keep the highest (match, fit) version |
| weave | 140 | $0.55 | retell the myth so it names its neighbours in their real directions; write a star-hop "how to find it" | name uniqueness, alphabet |
| epic | 6 + 7 | $0.18 | 6 composers from different narrators; 7 jurors score all six | consecutive stops ≤ 60° apart |
| final blind | 140 + 140 | $0.39 | blind read + judge of the finished set, for the numbers below | — |

Children run as `claude -p --model claude-haiku-5-5 --json-schema … --setting-sources "" --strict-mcp-config --tools ""` (or `--tools Read` for the agents that look at a PNG) from a scratch cwd, up to 28 at a time ([`fan.py`](fan.py)). Each output is a file, so every stage resumes after an interruption. Wall clock for the full build was about 40 minutes. Summed model time was 13.5 h.

## Results

- **Blind recognition went from 48/140 to 96/140.** A figure counts as recognised when the judge scores the blind guess at least 4 of 5 against the intended subject. The first critique round moved it to 62, the second to 64, and keeping the best-judged version of each figure across rounds gave 82. Best-of-3 fresh redraws for the remaining 58 gave 96. Most of the gain came from selecting among versions; a single redraw often scored worse than the draft it replaced, and the merge step originally kept it anyway.
- Both the blind reader and the judge are Haiku. "Recognised" therefore means Haiku-recognisable at the rendered scale with stars, lines and outline. No person has scored the drawings.
- The y axis decides whether a figure stands upright. In the pilot the frame was given as y-up, and Haiku drew the "Notched Doorpost" upside down: it put the foot at the top. In an SVG-style y-down frame the pilot figures came out the right way up.
- Haiku's first outlines retraced the stick lines. The fix was in the prompt: an outline is a silhouette with body and volume, 60–200 points, and a reader should be able to name the object from the drawing alone.
- The epic jury gave Aun's first-person confession 9/10 from all six jurors who scored it (one juror's keys did not parse). The other epics averaged between 5.3 and 8.2. The tour follows that epic.
- The weave pass resolved all 11 duplicate Ilvari names. It also returned some names wrapped in markdown or with the gloss attached (`*Lenkat* — "the Lidded Jar"`). `build.clean_name` keeps the bare word when it passes the alphabet check and otherwise falls back to the original name.

## Prior art

Two searches found [CANIS](https://portneuf.cose.isu.edu/courses/6673/papers/sewell.pdf), a computational-creativity system that invents constellations and names from star images. They also found Visual Cinnamon's [generated star maps](https://www.visualcinnamon.com/art/axiom-space/), which follow constellation-drawing rules such as no crossing lines. Neither builds a whole culture's sky with myths over the real catalogue, and neither runs a blind-recognition loop on the figures. That is two searches, not a survey.

## Rebuild

```bash
python3 prep.py                       # 140 regions, local gnomonic frames (work/regions.json)
python3 phase0_seeds.py && python3 fan.py work/t0.jsonl work/out0
python3 phase1_draw.py tasks && python3 fan.py work/t1.jsonl work/out1 && python3 phase1_draw.py check
./critic_loop.sh                      # 2 rounds blind → judge → redraw, then a blind pass
python3 phase2b_bestof.py select && python3 phase2b_bestof.py redraw 3 && ./bestof.sh
python3 phase3_weave.py weave work/out1 && python3 fan.py work/t3.jsonl work/out3
python3 phase3_weave.py epic work/out1 && python3 fan.py work/t4.jsonl work/out4
python3 phase3_weave.py jury work/out1 && python3 fan.py work/t4j.jsonl work/out4j && python3 phase3_weave.py pick work/out1
python3 phase2_critic.py judge work/art2 3 && python3 fan.py work/t2j_3.jsonl work/out2j_3
python3 build.py && python3 assemble.py   # work/sky.json → second-sky.html
```

`build.py` self-tests the inverse projection from drawing frame to sky. The worst error over all 2,061 stars is 0.02°.

Star data: [d3-celestial](https://github.com/ofrohn/d3-celestial) (BSD-3).
