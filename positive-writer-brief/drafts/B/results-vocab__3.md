## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text, but only when the pre-tokenizer stops splitting biomedical terms. A 50K-token BPE retrained on 39,000 PubMed abstracts, run with whitespace-only pre-tokenization, uses 21% fewer tokens than Ettin's tokenizer. The test set was 2,000 held-out abstracts, where Ettin's tokenizer spends 376 tokens per abstract.

The pre-tokenizer accounts for most of the gain. Ettin uses the GPT-2 pre-tokenizer, which cuts terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs. The retrained table then cannot learn those terms as single tokens. With the GPT-2 pre-tokenizer kept, the same retrained 50K table saves only 8%. Retraining the vocabulary on its own is therefore not enough to get the saving.

No published clinical ModernBERT has tried this. BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged.

Oskar asked the question on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment measures only tokenizer efficiency. Training models on the new table is the job of a separate follow-up experiment, `lifesci-ettin`.
