# Subagent prompts

All judges: Agent tool, `general-purpose`, model `haiku`. Paths shortened here to `exp/`.

## Shared scoring block (every arm)

```
- 3 = directly answers or explains the core of the question
- 2 = substantially relevant; covers a key part of the answer
- 1 = on-topic but does not help answer the question
- 0 = irrelevant
- Judge by reading the passages yourself. Do not write scripts or keyword heuristics.
```

## Arm-specific lines

| arm | input | extra instruction |
|---|---|---|
| A raw | `inputs/batchK.txt` | "Score directly. Do not write any analysis or reasoning." |
| B criteria brief | `inputs/batchK.txt` + `briefs_criteria.json` | "a relevance brief per qid, written by the lead researcher. It says what the asker wants and what a relevant passage contains. Treat it as the definition of relevance for that query." + score directly |
| C per-batch reasoning | `inputs/batchK.txt` | "Before scoring its passages, write an analysis of 80-150 words: what the question is really asking, what underlying concept or mechanism would answer it, and what a relevant passage would need to contain." Output adds an `analysis` field |
| E answer-guess brief | `inputs/batchK.txt` + `briefs_guess.json` | "a note per qid from the lead researcher giving the likely answer to the question. Use it to judge which passages help answer the question." + score directly |
| D single pass | `inputs/fullG.txt` (30 passages per query) + `briefs_criteria.json` | as B |

Output shape: `{"scores": {"<qid>": {"p01": <int>, ...}}}` written to `outputs/<ARM>_<K>.json`; reply limited to two lines.

## Run-2 spec (A2, B2, C2, and the E_2/E_3 reruns)

Inputs are the half files `inputs/batchKa.txt` + `inputs/batchKb.txt`, each small enough for one Read call, and the prompt adds:

```
How to read:
- Use the Read tool only. Each file fits in a single Read call with no offset or limit.
- Do not use Bash, Python, or any script. Every score must come from your own reading of that passage's text.
- Read every passage before scoring it.
```
