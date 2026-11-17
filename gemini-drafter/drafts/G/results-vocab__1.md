## Answer

A 50K-token BPE vocabulary retrained on 39,000 PubMed abstracts cuts Ettin's token count on held-out PubMed abstracts by 21%, provided it uses whitespace-only pre-tokenization. On 2,000 held-out PubMed abstracts, Ettin's default BPE tokenizer spends 376 tokens per abstract.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" Keeping the default GPT-2 pre-tokenizer caps the gain at 8%, because it splits biomedical terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. An initial round that kept the GPT-2 pre-tokenizer led to the assumption that a domain vocabulary does not help much; switching to whitespace-only pre-tokenization yields the 21% reduction.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this. Training models on the new token table is outside this experiment and moves to `lifesci-ettin`.
