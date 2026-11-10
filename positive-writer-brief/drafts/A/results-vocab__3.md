## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text, but only if the pre-tokenizer changes along with the table. A 50K-token BPE retrained on 39,000 PubMed abstracts and run with whitespace-only pre-tokenization uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts. Ettin's tokenizer averages 376 tokens per abstract on that set.

Retraining the table and keeping Ettin's GPT-2 pre-tokenizer saves only 8%. The pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so the new table never sees those terms whole and cannot learn them as single tokens. Round 1 of this experiment measured the 8% figure and concluded that domain vocabulary does not help much. Round 2 traced the shortfall to the pre-tokenizer and replaces that conclusion.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged. No published clinical ModernBERT has tried a domain token table, with or without a different pre-tokenizer.

The experiment started from Oskar's question on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment answers the token-table half. Training models on the new table is the job of a separate follow-up experiment, `lifesci-ettin`.
