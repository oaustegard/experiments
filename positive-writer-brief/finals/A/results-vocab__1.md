## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text if the pre-tokenizer changes along with the token table. On 2,000 held-out PubMed abstracts, Ettin's tokenizer spends 376 tokens per abstract. A 50K-token BPE retrained on 39,000 abstracts, with whitespace-only pre-tokenization, uses 21% fewer.

If Ettin's GPT-2 pre-tokenizer stays and only the table is retrained, the saving is 8%. The GPT-2 pre-tokenizer splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs. Round 1 measured only that configuration and concluded that domain vocabulary does not help much. Round 2 traced the small saving to the pre-tokenizer, so that conclusion no longer holds.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged. Neither has tried a domain token table, and so no published clinical ModernBERT has either.

On 2026-11-06 Oskar asked: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment answers the tokenizer half of that question. A follow-up experiment, lifesci-ettin, trains models on the new table.
