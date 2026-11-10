## Answer

A life-sciences vocabulary makes Ettin about a fifth cheaper on PubMed text, but only if the pre-tokenizer changes along with the token table. On 2,000 held-out PubMed abstracts, Ettin's BPE tokenizer spends 376 tokens per abstract. A 50K-token BPE retrained on 39,000 abstracts, with whitespace-only pre-tokenization, saves 21%.

The same retrained table saves only 8% when it keeps Ettin's GPT-2 pre-tokenizer. That pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so the new vocabulary never sees those terms whole. Round 1 measured the 8% and concluded that domain vocabulary does not help much. Round 2 traced the gap to the pre-tokenizer, which changes that conclusion.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried a domain token table.

The question came from Oskar on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment answers the tokenizer half of it. Training models on the new table is the follow-up experiment, lifesci-ettin.
