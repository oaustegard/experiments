## Answer

A life-sciences vocabulary makes Ettin 21% cheaper on PubMed text, as long as the tokenizer pre-tokenizes on whitespace only. On 2,000 held-out PubMed abstracts, Ettin's BPE tokenizer spends 376 tokens per abstract. A 50K-token BPE vocabulary retrained on 39,000 abstracts cuts that by 21% with whitespace-only pre-tokenization. With the GPT-2 pre-tokenizer it saves only 8%, because that pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs.

The question came from Oskar on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" Round 1 kept the GPT-2 pre-tokenizer, measured the 8%, and concluded that a domain vocabulary does not help much. Round 2 traced the small gain to the pre-tokenizer and measured the 21%.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried a domain vocabulary. Training models on the new table is outside this experiment. That work is in `lifesci-ettin`.
