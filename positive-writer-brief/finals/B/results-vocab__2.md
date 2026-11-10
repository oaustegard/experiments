## Answer

Yes, a life-sciences vocabulary makes Ettin cheaper on PubMed text, provided the pre-tokenizer changes with it. On 2,000 held-out PubMed abstracts, Ettin's tokenizer spends 376 tokens per abstract. A 50K-token BPE table retrained on 39,000 PubMed abstracts uses 21% fewer tokens on the same set when the pre-tokenizer splits on whitespace alone.

With the GPT-2 pre-tokenizer that Ettin uses, the same retrained table saves only 8%. The GPT-2 pre-tokenizer splits terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so BPE never sees those terms whole and cannot learn them as single tokens.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried a retrained table.

Oskar asked on 2026-11-06 whether "we [could] create an entirely custom domain Ettin model with custom token table ... for life sciences". This experiment answers the token-table half of that question. Training models on the new table is the subject of a separate experiment, lifesci-ettin.
