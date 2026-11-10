## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text, provided the pre-tokenizer changes along with the token table. A 50K-token BPE retrained on 39,000 PubMed abstracts, with whitespace-only pre-tokenization, uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts. Ettin's tokenizer spends 376 tokens per abstract.

Ettin's GPT-2 pre-tokenizer limits the gain. It splits terms like `GLP-1` and `IL-6` at hyphens and digits before BPE runs, so a retrained table never gets to learn them as single tokens. If the GPT-2 pre-tokenizer stays and only the table is retrained, the saving is 8%. Round 1 measured only that configuration and concluded that domain vocabulary does not help much. Round 2 traced the small saving to the pre-tokenizer, and that conclusion no longer holds.

BioClinical ModernBERT and Clinical ModernBERT ship Ettin's tokenizer unchanged. No published clinical ModernBERT has tried a domain token table.

Oskar asked on 2026-11-06 whether "we [could] create an entirely custom domain Ettin model with custom token table ... for life sciences". This experiment answers the tokenizer half of that question. The follow-up experiment, lifesci-ettin, trains models on the new table, and this experiment does not cover it.
