# regulator

**The Regulator** is a fair-play murder mystery set in a watchmaker's house in La
Chaux-de-Fonds on 31 December 1899. Every clock and watch in the house keeps its own
wrong time, one of six witnesses lies, and the dead man's minute repeater sounded the
true time as he fell. You have to name the murderer and the true minute of the blow.

- Live: **https://muninn.austegard.com/scratch/regulator/**. The source is
  [`app.html`](app.html) plus `work/case.json`. `python3 assemble.py` writes the artifact
  body `regulator.html`, and `python3 assemble.py --site DIR` writes the hosted page.
- The page has the dossier (the inspector's note, a cross-section of the house, six
  interviews), a bench that converts every reading to true time from the clock errors you
  set and lists the disagreements, an accusation form, and a reveal with the proof.

The aim was a small thing made exactly, after [`second-sky`](../second-sky/), which was a
large thing made by 1,843 calls. This one used about 700 calls, most of them checks.

## How the case is made

1. **A formal case** ([`world.py`](world.py)). The ground truth is a minute-by-minute
   evening and eight timepieces with fixed errors. Every statement a witness makes is a
   list of atoms: a reading of a named clock at a named moment (`r = t + offset`), a true
   time (the temple bell), an order or gap between moments, or a person in a room over an
   interval. The doctor's one lie is a single watch reading. Rosa's belief that her clock
   is a quarter-hour fast is recorded as a belief and carries no constraint.
2. **Proof** ([`verify.py`](verify.py), z3). It checks that the ground truth satisfies every
   honest statement and breaks the lie, and that exactly one suspect can be the murderer.
   The murderer stood in the atelier when the repeater sounded, and the solver drops all of
   that suspect's statements. It also checks that the attack time is unique (22:38) and that
   all seven clock errors that matter are pinned. For each innocent it enumerates minimal
   unsatisfiable subsets: the statements that cannot all be true if that person did it.
   Every one has at least five statements.
3. **Interviews** ([`interviews.py`](interviews.py), [`choose.py`](choose.py),
   [`run_interviews.sh`](run_interviews.sh)). Haiku 5.5 writes five drafts per witness from
   first-person fact cards ([`facts.py`](facts.py)) and a one-page world
   ([`bible.md`](bible.md)). Each draft gets two checks:
   - a blind extractor that never sees the facts lists every `(timepiece, reading)` pair,
     and code compares the set with the expected one;
   - a judge holding the facts marks each one exact, altered or missing, and lists
     additions that would give a detective new information about times, order or
     whereabouts.

   Failures get one repair. Survivors face two adversarial ambiguity hunters each and a
   five-juror ranking. The best clean draft per witness is edited by hand
   ([`final/`](final/)), then [`finalcheck.py`](finalcheck.py) runs extractor, judge and
   hunter three times each on the edited text.
4. **Detectives.** 20 Haiku 5.5, 5 Sonnet 5.5 and 3 Opus 5.5 read the whole dossier once
   and answer.

## Results

| version | what changed | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| v1 | first case | 10/14 | 5/5 | – |
| v2 | tightened prose, fewer giveaways | 19/20 | 5/5 | – |
| v3 | most direct bell calibrations removed; Rosa's mistaken belief added | 9/19 | 3/5 | 3/3 |

"Right" means both the doctor and 22:38. On v3, 4 Haiku detectives named Bastien and 5 more named the doctor at the wrong minute (22:47 or 22:48); a 20th Haiku was still working after 15 minutes and was stopped. All three Opus detectives gave the intended chain: regulator → hall clock → Rosa's clock → piano → the doctor's watch.

- **The first versions were arithmetic.** Every witness had read a clock against the temple
  bell, and the doctor's departure time contradicted three other accounts after one
  subtraction. Haiku solved v2 in about a minute per detective.
- **v3 chains the calibrations.** Rosa's clock is calibrated through the hall clock's strike
  of eleven. The hall clock is calibrated through Lina's regulator. Bastien's watch is
  calibrated through the hall clock as he came in. A solver who takes Rosa's word that her
  clock is a quarter-hour fast finds the piano starting at 22:43, the minute the doctor
  claims he left. The doctor's story then holds, and Bastien's watch looks like the lie.
  Several Haiku detectives named Bastien, which is the answer the trap is built to produce.
  Sonnet detectives that fell for it kept the doctor but put the blow at 22:48. Haiku took
  four to seven minutes per detective on v3.
- **The judge prompt had to be narrowed.** The first fidelity judge flagged every sentence
  of texture (Chopin, the quarrel at dinner) as an added claim. The fix asked only for
  information a detective could use to place a time, an order or a person.
- **Re-verify after hand edits.** The triple check on the final text found a bug I had
  introduced: a reordered answer left the visitor as an unnamed "he" that read as the dead
  man. It also found Haiku inventions that one judge pass had missed: a library mantel clock
  nobody else mentions, and a line implying Bastien walked away from the windows.
- **Opinion versus observation is the fair-play device.** Rule 2 says honest witnesses report
  what they saw and heard, "what any of them believes about a clock is another matter." Two
  observations flag the trap: Lina saw the master with Rosa's clock that afternoon, and with
  a quarter-hour offset Lina takes twelve minutes to walk downstairs.

## Prior art

Three searches turned up the components. [SATBench](https://arxiv.org/abs/2505.14615)
generates logic puzzles from SAT formulas and has an LLM write the story.
[MuSR](https://arxiv.org/abs/2310.16049) builds murder mysteries from a reasoning tree with
fact-consistency checks. [Daily Murder](https://www.producthunt.com/products/daily-murder/makers)
verifies one solution per puzzle. Christie's *The Clocks* and a few games use a wrong clock
as a clue. I did not find, in three searches, a clock-calibration mystery with
solver-proven uniqueness, enumerated refutations per suspect, round-trip-verified prose and
a working calibration bench.

## Rebuild

```bash
python3 verify.py                 # proof; writes work/verify.json
./run_interviews.sh               # drafts -> checks -> repair -> hunt + jury -> work/chosen.json
# hand edits into final/*.md, then:
python3 finalcheck.py tasks && python3 fan.py work/t_final.jsonl work/fin && python3 finalcheck.py report
python3 fan.py work/t_det.jsonl work/det --conc 25      # detectives
python3 stats.py && python3 build.py && python3 assemble.py
```

`fan.py` is second-sky's headless `claude -p` fan-out, one JSON file per output, resumable.
`work_v2/` holds the v1 and v2 runs.
