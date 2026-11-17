## Answer

On 2,000 held-out PubMed abstracts, a 50K-token BPE vocabulary retrained on 39,000 PubMed abstracts uses 21% fewer tokens than Ettin's stock BPE tokenizer, which spends 376 tokens per abstract. The 21% depends on replacing the GPT-2 pre-tokenizer with whitespace-only pre-tokenization.

Oskar asked on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" The first pass kept the GPT-2 pre-tokenizer. With it, the retrained vocabulary saved 8%, and we read that as domain vocabulary not helping much. The GPT-2 pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so the retrained vocabulary never sees them whole. With whitespace-only pre-tokenization, the same 50K table saves 21%.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried a domain vocabulary. Training models on the new token table is outside this experiment; the follow-up experiment `lifesci-ettin` does it.
