## Answer

A 50K-token BPE vocabulary retrained on 39,000 PubMed abstracts uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts, provided it uses whitespace-only pre-tokenization. Ettin's stock tokenizer spends 376 tokens per abstract on that set.

This answers Oskar's question from 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?"

With Ettin's GPT-2 pre-tokenizer kept in place, the same retrained table saves only 8%. That pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so the new vocabulary never sees them whole. After the first run, which used the GPT-2 pre-tokenizer, I concluded that a domain vocabulary helped little. That conclusion was wrong: the pre-tokenizer was the limit, not the vocabulary.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this change. The `lifesci-ettin` follow-up experiment trains models on the new token table; that work is outside this experiment.
