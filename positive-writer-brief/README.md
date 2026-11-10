# positive-writer-brief

Muninn writes public prose (READMEs, PR bodies, blog sections, results write-ups, commit messages, Bluesky posts) with two documents loaded before the first word: `muninn-voice-signature`, which puts a "Scan for: <tell>" list under every practice, and the `declauding` skill, which tells the writer to load it before drafting and catalogues the tics it removes. So the writer drafts with the banned phrases in context.

This experiment asks whether a drafting brief written as "do X, because the reader Y", with none of those phrases in it, gives better first drafts, and whether it still helps once the declauding editor pass has run on the draft.

The design is in Muninn memory 419fdbdc. It rests on caps-emphasis (memory 28b289c1, [`../caps-emphasis/`](../caps-emphasis/RESULTS.md)): on a 321M model, a directive that named a forbidden word raised that word's probability by about 2.5 log-odds, and a neutral mention of the word gave about 40% of that effect. Whether this carries over to an Opus-class drafter and to stylistic constructions is what arm A against arm B measures.

## Arms

| arm | drafter context | editor pass |
|---|---|---|
| A | voice signature + declauding `SKILL.md` | none |
| B | [`brief_b.md`](brief_b.md) only | none |
| A-final | as A | declauding pass |
| C | as B | declauding pass |

- **Brief B** converts the voice signature's positive markers and practices, plus declauding's drafting-time moves (step 1a, state the one point first; the newcomer read; describe the end state rather than the change, memory 3f358ed4; the outline test from memory 43c3d7ae), into do-because statements. `python3 check_brief.py` checks it against all 160 regexes in declaude_lint's RULES table and the 49 quoted tell strings in the voice signature's Scan-for lists, and exits 1 on any hit.
- **Drafter**: `claude-opus-5-5` at effort low (the effort Muninn's sessions run at), as headless `claude -p` children with no tools, a system prompt holding the arm's documents and a Muninn persona line, and the task as the prompt. Four drafts per task per arm.
- **Editor pass**: `claude-opus-5-5` at effort low, system prompt holding declauding `SKILL.md`, `references/register.md`, `references/exceptions.md` and the voice signature; prompt holding the task, the facts, the draft and the output of `declaude_lint.py` on it. It runs workflow steps 1 to 4 and 6. The scripted stages that need a model or a subagent (2a ranker, 2b structural review, 4a newcomer read, 5 diff) are not run; the editor is told to do their checks by eye. The same editor runs on A and B drafts.
- **Tasks**: twelve, two per surface, in [`tasks.json`](tasks.json), each with its facts in the prompt. Six are built to invite focus failures rather than sentence tics: four hand the writer the facts as a changelog or a round-by-round log and ask for text that describes the thing (two READMEs, two results Answers), and two hand over the facts as the author's own sequence of attempts and feelings and ask for a blog section. Facts for the Answer, blog and Bluesky tasks come from real experiments in this repo; the README, PR and commit facts are partly invented fixtures.
- Snapshots of everything the children saw are in `context/` (declauding 0.13.2, voice signature as of 2026-11-10). Neither the skill nor the config is changed by this experiment.

## Metrics

1. **Lint rate**: hits from `context/declaude_lint.py` per 1,000 words of the text, counting distinct (line, column) positions and leaving out the document-level density notes.
2. **Blind judge**: Opus 5.5 and Sonnet 5.5, at effort medium, each rate every text 1 to 7 on register and on focus, and count invented claims. One call per task per stage holds all eight texts (four A, four B; or four A-final, four C), shuffled, under random ids, with no arm labels. A text's score is the mean of the two judges.
3. **Newcomer read**: per declauding `references/fresh-eyes.md`, a Sonnet 5.5 reader at effort medium who is told they are the task's reader and have never seen the project lists every phrase that would stop or mislead them. The score is the number of stumbles.

Differences are B − A on raw drafts and C − A-final on finished text, averaged over the twelve tasks. 95% CIs come from a two-level bootstrap (resample tasks, then the four drafts of each arm within a task), 10,000 resamples, seed 7.

## Pre-registered predictions

Written and pushed before any scored draft was generated. A two-draft pilot checked the plumbing only, and its drafts are not in the data.

- **P1** (the priming question): B has fewer lint hits per 1,000 words than A on raw drafts. Holds if the 95% CI on B − A lies below 0.
- **P2**: B's focus on raw drafts is at least as good as A's. Holds if the lower 95% bound on B − A judge focus is above −0.5 on the 7-point scale.
- **P3**: after the editor pass, C has fewer lint hits per 1,000 words than A-final. Holds if the 95% CI on C − A-final lies below 0.
- **P4**: C's focus is at least as good as A-final's, with the same −0.5 margin as P2.
- **P5** (exploratory, no threshold): B drafts cause fewer newcomer stumbles than A drafts, most of all on the six focus-trap tasks.

## Reproducing

```bash
python3 check_brief.py                      # 0 tell strings, exit 0
python3 score.py                            # recomputes every number in RESULTS.md from drafts/, finals/, scores/
```

Regenerating the drafts and scores (about $15 in Claude Code usage):

```bash
S=/mnt/skills/user/dispatching-subagents/scripts/headless_fanout.py
for st in draft edit; do        # drafter and editor at effort low
  python3 pipeline.py prep $st
  python3 $S run runs/$st/tasks.jsonl --work runs/$st --effort low --concurrency 12
  python3 pipeline.py collect $st
done
for st in fresh judge; do       # readers and judges at effort medium
  python3 pipeline.py prep $st
  python3 $S run runs/$st/tasks.jsonl --work runs/$st --effort medium --concurrency 12
  python3 pipeline.py collect $st
done
```

Results: [`RESULTS.md`](RESULTS.md).
