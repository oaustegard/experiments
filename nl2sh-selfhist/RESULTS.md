# nl2sh-selfhist — a corpus and an eval where neither side is Claude's

`nl2sh-scoping` and `nl2sh-retrieval` ran on NL2Bash, whose 60% `find` skew
distorted two measurements and whose English was written by annotators looking
at the commands. This directory replaces both halves of that eval and reports
the number that results.

The corpus story — this session's history (unusable as a benchmark), the
Zenodo/UCI cybersecurity-training set (16,065 real commands, constant prior
0.189 vs NL2Bash's 0.603), documentation-coverage on real usage, and the tldr
alias-page bug — is in the git history of this directory and in `corpus_probe.py`.
The headline is the eval those pieces enabled.

## The eval neither side authored

The cyber corpus supplies real commands but no natural language. `gen_nl.py`
closes that with the one independent model available: **Gemini writes the
request for each command**, instructed to describe the intent without naming the
utility. So the commands come from real users, the NL from an independent model,
and **nothing in the eval was written by the model under test**. Of 38 requests,
4 named the utility anyway; the leak-free 34 are the ones quoted.

`run_independent_eval.py` routes each request through a fine-tuned model with the
gold utility's tldr example plus two distractors in context — the same k=3 setup
the gate uses.

## The result: 0.62, against the gate's 0.92

| model | utility routing (leak-free, n=34) | all (n=38) | command rate |
|---|---|---|---|
| fine-tuned **RAG** | **0.618** | 0.658 | 0.842 |
| fine-tuned **non-RAG** | 0.529 | 0.579 | 0.895 |

**The NL2Bash gate's 0.923 was inflated.** The same RAG model that scored 0.923
on the gate's non-`find` slice scores **0.618** here — on independent phrasing,
over a real command distribution, with no utility names leaked. That 0.30 drop
is the cost of every flattering property the gate had: templated phrasing from
one author, a distribution 27/40 `find`, and (34.7% of the time) the answer
named in the request. This is the honest capability of a 350M model fine-tuned
on 600 rows: **it routes about three real requests in five.**

**The RAG base's edge widened, and stayed inside the confound.** 0.618 vs 0.529
is a clearer margin than the gate's one-example difference, but the non-RAG
sibling still lacks the special tokens (18 vs 5 tokens per source block), so the
gap remains model-training *and* delimiters, not RAG training alone.

## How it fails

Of the 13 leak-free misses, the modes split two ways:

* **Hallucinating a distractor's utility** — *"recover the password for
  invoices2019.zip"* produced `pgmbentley`, *"crack the password hashes"*
  produced `calligrastage`, *"open authorized_keys in an editor"* produced
  `b4-am`. None of these are real commands; they are corpus utilities the model
  reached for over the gold one in its context.
* **Abstaining** — *"show all files including hidden"* (gold `ls -a`) and
  *"listen on port 51071"* (gold `nc`) produced no command at all.

A handful are defensible semantic near-misses — *"find the location of
rockyou.txt.gz"* routed to `find … -exec grep` instead of `locate` — which
`utility_ok` scores as wrong but a functional-equivalence judge might not.

By tier the leak-free numbers are head 7/13, mid 7/12, tail 7/9; the tail
scoring highest is small-n noise, driven by security tools whose independent NL
was distinctive.

## What this settles for the whole line of work

The eval-authorship problem that ran through `gh-mcp-regex-fit` (an 89-row set
Claude wrote) and `nl2sh-retrieval` (NL2Bash's annotator-written English) is
fixed here, and the number moved a lot: **0.92 → 0.62** when neither side is the
model's own. Any figure in this thread measured against a self-authored or
utility-naming eval should be read as an upper bound, not a capability.

The 0.618 is still not scored on functional equivalence — it is utility routing.
The `funceq.py` harness exists but most cyber commands touch the network or
security tooling and will not run in its fixture, so a functional number on this
corpus needs a broader sandbox than this container provides. That is the next
measurement, and it will be lower again.

## Reproduce

```bash
# corpus
curl -sL -o data.zip "https://zenodo.org/api/records/8136017/files/data.zip/content"
unzip -q data.zip -d cyber
python3 corpus_probe.py --cyber cyber --chunks ../nl2sh-retrieval/data/chunks.jsonl --tldr <tldr>/pages

# the independent eval
python3 gen_nl.py --sample cyber_sample.json            # Gemini writes the NL (gateway)
python3 run_independent_eval.py --model ../nl2sh-retrieval/ft     --out results_independent_rag.json
python3 run_independent_eval.py --model ../nl2sh-retrieval/ft_nonrag --out results_independent_nonrag.json
```

`cyber_nl.json` (the generated NL) is committed; the cyber corpus and model
weights are gitignored.
