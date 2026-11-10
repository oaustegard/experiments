## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text, provided the pre-tokenizer changes along with the token table. A 50K-token BPE retrained on 39,000 PubMed abstracts and run with whitespace-only pre-tokenization uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts. Ettin's tokenizer spends 376 tokens per abstract on that set.

Retraining the table alone saves 8%. The GPT-2 pre-tokenizer that Ettin uses splits terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs. BPE therefore never sees those terms whole and cannot learn them as single tokens. The first round of this experiment kept that pre-tokenizer and concluded that domain vocabulary does not help much. That conclusion came from the pre-tokenizer, not from the vocabulary.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged. As far as this experiment found, no published clinical ModernBERT has used a domain token table.

This experiment measures token counts only. It does not show whether a model trained on the new table matches Ettin's accuracy. That is the subject of the follow-up experiment, lifesci-ettin, which trains models on the new table. The experiment answers Oskar's question of 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?"
