# Prior art

Searched before writing the analysis, and again while it ran. Four of the five
questions in the issue turned out to have some published answer; one did not.

## Directly on target

**[Attention is Case-Sensitive](https://arxiv.org/abs/2608.03711)** (Dillitzer,
Sohn, Corso, Auerbach; ECCV 2026, posted 2026-08-04). Thirteen models, nine
LLMs from 1B to 14B plus four vision-language models, measured on MMLU-Pro,
ARC-C, SQuADv2, RefCOCOg, XQuAD and HumanEval. Uppercasing a span against
lowercase context concentrates attention on it by 2.06 percentage points, and
alternating case by 2.77. The attention gain does not buy accuracy: alternating
case costs 2.88 points of accuracy while pulling the most attention.
Their result that matters most here is that **reasoning models show near-zero
case sensitivity**, which they attribute to the think phase. Baguettotron is a
reasoning model. That paper predicts the null this experiment mostly found.

It does not run a dose-response, a markdown-bold arm, or a within-tokenizer
token-count control. Its argument that the effect is not a sequence-length
artifact rests on consistency across tokenizers, which is a different claim from
holding token count fixed inside one.

**[Semantic Gravity Wells: Why Negative Constraints Backfire](https://arxiv.org/abs/2601.08070)**
(Rana, 2026-01-12). Qwen2.5-7B-Instruct, 2,500 single-word prohibitions × 16
samples. Violation probability is logistic in the model's intrinsic probability
of the forbidden token: `p(violation) = σ(−2.40 + 2.27·P₀)`, R² 0.78. Suppression
is 4.4× weaker in the cases that fail. 87.5% of failures show a priming
signature, attending to the forbidden word's mention more than to the negation.
This is why the items here are stratified by baseline pressure and why the dose
sweep freezes the forbidden word's case.

**[Don't Think of the White Bear](https://arxiv.org/abs/2511.12381)** (Mann et
al., 2025-11-15). ReboundBench, 5,000 negation prompts, nine models. Uses the
same teacher-forced instrument this experiment uses, and finds the same ironic
rebound: telling a model not to say X raises P(X). It states plainly that it does
not validate the logprob measure against generation, which is the gap the
generation arm here is for.

**[FormatSpread](https://arxiv.org/abs/2310.11324)** (Sclar, Choi, Tsvetkov,
Suhr; ICLR 2024). 320 tasks. Prompt formatting is worth up to 76 accuracy
points, but casing specifically is close to null: median casing spread 0.188,
with 0% of tasks strongly different between casing variants. The largest
casing-specific measurement that existed before this one, and it already
disagreed with the folklore.

**[Spotlight Your Instructions](https://arxiv.org/abs/2505.12025)**
(Venkateswaran & Contractor, IBM). Supplies the attention metric used here,
ψ(i) = Σ_{j∈S}A_ij / Σ_k A_ik, and shows attention deficit on an instruction
span causes instruction-following failure — forcing ψ to 0.1 buys 26% on IFEval.

**[UPPERCASE IS ALL YOU NEED](https://github.com/ASSERT-KTH/UPPERCASE_IS_ALL_YOU_NEED)**
(SIGBOVIK 2025). A joke venue, and the only prior head-to-head caps A/B on a real
benchmark: 163 HumanEval docstrings, ~87% pass rate either way. Null.

## On the attention half

Attention weight is not causal importance —
[Jain & Wallace 2019](https://aclanthology.org/N19-1357/),
[Wiegreffe & Pinter 2019](https://aclanthology.org/D19-1002.pdf),
and the [faithfulness violation test](https://arxiv.org/abs/2201.12114). High
attention can even mark suppression. Correlating attention mass with case would
not have established anything, which is why there is a knockout arm.

Separately, the prompt-injection literature
([TopicAttack](https://arxiv.org/abs/2507.13686)) finds that repetition and token
volume raise attention mass on their own, with content held constant. That
contradicts the ECCV paper's dismissal of length and is the reason the
token-count control here is within-tokenizer rather than across.

## Folklore

[Anthropic's prompt engineering guidance](https://claude.com/blog/best-practices-for-prompt-engineering)
recommends minimal formatting and no reliance on caps. Assertion, no data. The
advice-post genre around capitalised emphasis is large and, as far as four
searches found, entirely unmeasured.

## Not found

Did not find, in the searches run: any dose-response on the fraction of a prompt
capitalised; markdown bold used as a token-count-matched, case-constant control;
a case effect regressed on tokenizer-induced token-count delta inside one
tokenizer; or a capitalisation base rate measured in a synthetic pretraining
corpus. Those are where this experiment is not a replication. Four queries is
not a survey — this says "did not find it", not "it does not exist".
