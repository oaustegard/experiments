## Answer

A 50K-token BPE table trained on 39,000 PubMed abstracts cuts Ettin's token count on held-out abstracts by 21% when paired with whitespace-only pre-tokenization. Ettin's default BPE tokenizer spends 376 tokens per abstract across 2,000 held-out PubMed abstracts.

The GPT-2 pre-tokenizer controls most of the savings because it splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. In my first round I kept the GPT-2 pre-tokenizer and concluded that domain vocabulary did not help much, because the retrained 50K table saved 8%. Replacing that pre-tokenizer with whitespace splitting raises the reduction to 21%.

On 2026-11-06 Oskar asked: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this. A follow-up experiment, `lifesci-ettin`, trains models on the new token table.
