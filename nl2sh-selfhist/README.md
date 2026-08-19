# nl2sh-selfhist — where a real evaluation corpus for a shell helper comes from

Three passes of `gh-mcp-regex-fit` and `nl2sh-retrieval` ran on **NL2Bash**,
whose **60.3% `find` skew** distorted two separate measurements: a
query-independent frequency prior beat the retrieval headline (0.625@1 against
the real system's 0.098), and a constant *"always answer `find`"* scored **0.675**
on the model gate. Both were caught only by an adversarial pass. This directory
looks for a corpus that does not have that problem.

## Two candidates, measured

### 1. This session's own shell history — sound idea, does not work

289 Bash calls issued tonight. Its distribution *is* flatter than NL2Bash — 22
distinct leading utilities with the commonest (`cd`) at **24.2%** against `find`
at 60.3% — which was the point of the suggestion and it holds. But it is not a
benchmark:

| filter | remaining |
|---|---|
| all Bash calls | 289 |
| single-line, no heredoc | 66 |
| general-shell (not invoking this project's scripts) | **26** |
| distinct task shapes among those | ~8 |

Roughly 15 of the 26 are variants of *"print lines X to Y of file F"*. **An
agent's shell history is file-slicing and project-script invocation, not general
utility composition.** Worse, the Bash tool's `description` field — which would
have supplied genuine paired natural language, the thing every corpus here
lacks — **is not persisted in the transcript**; only the commands survive.

Kept as an out-of-distribution probe (`selfhist_eval.py`, n=14, hand-authored in
those real task shapes and rewritten to run inside `funceq.py`'s fixture), not
as a benchmark.

### 2. A public corpus of real human shell sessions — this one works

The Zenodo/UCI dataset of hands-on cybersecurity training ([record
8136017](https://zenodo.org/record/8136017), UCI 869, **CC-BY-4.0**):
**16,065 bash commands from 275 participants**, with timestamps and working
directories, captured in real terminals.

| | NL2Bash | cyber-training | this session |
|---|---|---|---|
| commands | 12,607 | **16,065** | 289 |
| distinct leading utilities | 389 | **696** | 22 |
| **constant prior** (commonest utility) | **0.603** (`find`) | **0.189** (`ls`) | 0.242 (`cd`) |
| utilities appearing once | 176 | **366** | — |
| top-50 coverage | 88.1% | 87.0% | 100% |

**A 3.2× lower constant prior over a longer tail.** That is the instrument the
skew problem needed. Its own bias is *domain* rather than shape — `nmap`,
`fcrackzip`, `john` and `msfconsole` are over-represented because participants
were doing security exercises — and it carries **no natural-language pairing**,
so it fixes the distribution and not the annotation.

## What it immediately measured: documentation coverage on a real distribution

The retrieval tier reads a corpus of 31,169 tldr and man chunks over 4,698
utilities. Against 696 utilities people actually ran:

| slice | utilities covered | by invocation |
|---|---|---|
| all 696 | **24.4%** | **87.7%** |
| top 50 by usage | 84.0% | 96.6% |
| **tail (used once, n=366)** | **9.8%** | 9.8% |

Documentation covers **88% of what gets run and 24% of what exists** — the head
is well served and the tail is not, on real usage rather than a scraped corpus.

**And most of the uncovered tail is not documentable at all.** The
most-invoked utilities with no chunk fall into categories no corpus improvement
reaches:

| utility | invocations | why it is missing |
|---|---|---|
| `ll` | 114 | **shell alias** — exists in nobody's documentation |
| `python3` | 96 | tldr stub aliasing `python` (see bug below) |
| `sqlmap`, `msfdb` | 119 | genuine domain tools |
| `./ssh2john.py` | 40 | **local script path** |
| `mfsconsole` | 22 | **a typo for `msfconsole`** |
| `DEK-Info:` | 21 | **parse artifact** — pasted text, not a command |

So the retrieval tier's ceiling on real usage is lower than a naive coverage
number suggests, and the gap is aliases, local scripts and typos — which argues
for reading the user's shell configuration and `$PATH`, not for a bigger corpus.

## One corpus bug, found and quantified

`build_corpus.py` drops 756 tldr "stub" pages whose body is *"View documentation
for the original command"*. **379 of those are alias pages** — `whoami` is
documented as an alias of `id --user --name`, `python3` as an alias of `python`
— and 303 name a target that *is* in the corpus. Resolving the redirect instead
of dropping the page is nearly free.

It is also nearly worthless: coverage moves **24.4% → 25.6%** of utilities and
87.7% → 88.8% of invocations, because the recoverable aliases are obscure
(`llvm-lipo`, `pw-play`, `apparmor_status`) while the ones that matter — `ll`,
`dir` — are *shell* aliases with no tldr page at any name. Worth fixing; not
worth expecting anything from. `data/tldr_aliases.json` has the mapping.

## Reproduce

```bash
curl -sL -o data.zip "https://zenodo.org/api/records/8136017/files/data.zip/content"
unzip -q data.zip -d cyber
python3 corpus_probe.py --cyber cyber \
    --chunks ../nl2sh-retrieval/data/chunks.jsonl --tldr <tldr>/pages
```

`raw_commands.json`, `clean_commands.json` and `general_commands.json` are this
session's extracted history at each filter stage; `results_corpus.json` holds
the coverage table. The cyber corpus is not vendored — it is CC-BY-4.0 and
2.4 MB, so it is fetched.
