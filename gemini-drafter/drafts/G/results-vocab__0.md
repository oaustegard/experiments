## Answer

A 50K-token BPE vocabulary retrained on 39,000 PubMed abstracts cuts token count by 21% on 2,000 held-out PubMed abstracts when paired with whitespace-only pre-tokenization. Ettin's stock BPE tokenizer spends 376 tokens per abstract on that held-out set.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" Our first pass kept the GPT-2 pre-tokenizer and saved 8%, which looked like domain vocabulary did not help much. The GPT-2 pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so replacing it with whitespace-only pre-tokenization unlocks the 21% reduction.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this. Training models on the new token table moves to a follow-up experiment, `lifesci-ettin`.
