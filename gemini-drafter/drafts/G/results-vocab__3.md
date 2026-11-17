## Answer

A 50K-token BPE vocabulary trained on 39,000 PubMed abstracts cuts token count on PubMed text by 21% over Ettin's tokenizer when it uses whitespace-only pre-tokenization. Ettin's stock BPE tokenizer spends 376 tokens per abstract on a 2,000-abstract held-out set.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" Keeping Ettin's GPT-2 pre-tokenizer limits the same 50K retrained table to an 8% saving, because that pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. I had concluded after the first run that a domain vocabulary helped little; replacing the pre-tokenizer produced the 21% reduction.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this change. Training models on the new token table is the subject of the `lifesci-ettin` follow-up experiment.
