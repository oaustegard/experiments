## Answer

Yes, a life-sciences vocabulary makes Ettin cheaper on PubMed text. A 50K-token BPE table retrained on 39,000 PubMed abstracts uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts. That figure only holds when the pre-tokenizer splits on whitespace alone. Ettin's tokenizer spends 376 tokens per abstract on the same set.

The pre-tokenizer accounts for most of the difference. With the GPT-2 pre-tokenizer that Ettin uses, the retrained table saves only 8%. The GPT-2 pre-tokenizer splits terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so BPE never sees those terms whole and cannot learn them as single tokens. Retraining the vocabulary helps only if the pre-tokenization changes with it.

No published clinical ModernBERT has tried this. BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged.

Oskar asked on 2026-11-06 whether "we [could] create an entirely custom domain Ettin model with custom token table ... for life sciences". This experiment answers the token-table half of that question. Training models on the new table is the subject of a separate experiment, lifesci-ettin, and this experiment does not cover it.
