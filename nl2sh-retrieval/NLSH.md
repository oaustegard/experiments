# nlsh — the on-device terminal helper

The local path, assembled and working: a natural-language shell helper where
**nothing leaves the machine**. It trades accuracy and the cloud's safety layer
for privacy and offline operation — the tradeoff chosen deliberately over the
`gemini-3.5-flash-lite` cloud path (`ARCHITECTURE.md`), which is more accurate
(0.77 vs ~0.62) but sends every request to a third party.

## The pipeline, and why each stage is shaped the way it is

```
request ──► BM25 retrieval ──► fine-tuned 350M ──► parameter audit ──► confirm ──► run
            ($PATH-scoped,      (Pleias-350M,       (warn on dropped     (never
             top k=3)            ft/)                 literals)            auto-exec)
```

Every choice here is a measurement from this repo, not a preference:

- **Retrieval scoped to `$PATH`.** `nl2sh-selfhist` measured that restricting
  the 31k-chunk corpus to installed utilities nearly doubles recall@1. nlsh does
  it from `os.environ["PATH"]` at startup — free, and a helper knows what is
  installed.
- **Top k=3.** The model degrades past three sources (6/8 → 4/8 → worse in the
  gate). nlsh feeds exactly three, one example per utility.
- **The model recovers weak retrieval.** In testing, *"find files bigger than
  100MB"* retrieved `git`, `blkdiscard`, `find` — `find` only third — and the
  model still produced `find . -type f -size +100M -exec ls -l {} \;`. Retrieval
  narrows; the model decides.
- **The parameter audit does not splice, it warns.** `extract_params` pulls
  literals from the request, but only 53% of extractions are whole command
  tokens, so nlsh does not inject them. It checks the generated command *contains*
  each literal the request named, and warns when one is missing — catching the
  identifier-drop `monad-bsky` measured a small model make 49% of the time.
  Demonstrated: *"count the lines in notes.txt"* produced `tail -n 1 | wc -l`
  and nlsh warned *"request said filename 'notes.txt' — not in the command"*.
- **A confirmation gate, and a destructive-command guard.** The command is
  printed and never runs until you press enter; `rm -rf /`, `mkfs`, `dd of=/dev`,
  fork bombs are printed but nlsh refuses to run them. On the local path there is
  no upstream safety model — this gate is the only safety there is.
- **Graceful degradation.** With no model present nlsh runs retrieval-only,
  showing the closest documented utilities. A worse tool, honest about its mode.

## Honest state

It runs end to end on CPU, model loaded once (~13 s) then ~12 s per query
unquantised — usable in a REPL, sluggish one-shot; far faster on Metal/GPU or
quantised. **The ceiling is the model.** At ~0.62 independent-eval routing the
suggestions are frequently rough, and the value on a weak model is as much the
audit ("this dropped your filename") as the command itself. The scaffold is
done; making the local path genuinely good is a model problem — more training
data, more than one epoch, or a stronger sub-1B base — not a plumbing one.

## Use

```bash
python3 nlsh.py "find files bigger than 100MB under src"   # one-shot, then confirm
python3 nlsh.py                                            # REPL
python3 nlsh.py --explain "..."                            # show retrieval + audit, do not run
python3 nlsh.py --no-scope "..."                           # use the full corpus, not just $PATH
```

Needs `data/chunks.jsonl` (build with `build_corpus.py`) and `ft/` (train with
`finetune_gate.py`). Both are gitignored; the model weights are 1.4 GB.
