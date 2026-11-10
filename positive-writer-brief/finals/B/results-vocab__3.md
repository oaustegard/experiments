## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text, but only when the pre-tokenizer stops splitting biomedical terms. A 50K-token BPE retrained on 39,000 PubMed abstracts and run with whitespace-only pre-tokenization uses 21% fewer tokens than Ettin's tokenizer. The test set was 2,000 held-out abstracts, on which Ettin's tokenizer spends 376 tokens per abstract.

With the GPT-2 pre-tokenizer that Ettin uses, the same retrained 50K table saves only 8%. That pre-tokenizer cuts terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so most of the 21% comes from changing the pre-tokenizer, not from the new vocabulary alone.

BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged, so no published clinical ModernBERT has tried this.

Oskar asked the question on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment measures only tokenizer efficiency. A separate follow-up experiment, `lifesci-ettin`, trains models on the new table.
