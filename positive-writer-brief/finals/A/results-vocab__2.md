## Answer

Yes. A life-sciences vocabulary makes Ettin about a fifth cheaper on PubMed text, provided the pre-tokenizer changes along with the token table. On 2,000 held-out PubMed abstracts, Ettin's BPE tokenizer spends 376 tokens per abstract. A 50K-token BPE retrained on 39,000 abstracts, with whitespace-only pre-tokenization, saves 21%.

Retraining the table alone saves 8%. That was the round 1 result, and the reading at the time was that domain vocabulary does not help much. Round 2 found the cause in the GPT-2 pre-tokenizer, which splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. The retrained table could not learn those terms as single tokens, because they never reached it whole.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" No published clinical ModernBERT has tried a custom token table: BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged.

This experiment measures the token count only. Training models on the new table is the job of a follow-up experiment, `lifesci-ettin`.
