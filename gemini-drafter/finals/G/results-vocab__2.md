## Answer

On 2,000 held-out PubMed abstracts, Ettin's BPE tokenizer spends 376 tokens per abstract. A 50K-token BPE table retrained on 39,000 abstracts cuts that by 21% when it is paired with whitespace-only pre-tokenization. With Ettin's GPT-2 pre-tokenizer left in place, the same retrained table cuts it by 8%.

The gap comes from the GPT-2 pre-tokenizer. It splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. My first round kept that pre-tokenizer, measured the 8%, and concluded that domain vocabulary does not help much. The 21% from the second round replaces that reading.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried a domain token table. This experiment covers only the token table. A follow-up experiment, `lifesci-ettin`, trains models on it.
