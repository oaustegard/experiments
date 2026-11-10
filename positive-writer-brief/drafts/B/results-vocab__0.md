## Answer

Yes. A life-sciences vocabulary makes Ettin cheaper on PubMed text if the pre-tokenizer changes along with it. A 50K-token BPE table retrained on 39,000 PubMed abstracts, with whitespace-only pre-tokenization, uses 21% fewer tokens than Ettin's tokenizer on 2,000 held-out abstracts. Ettin's tokenizer averages 376 tokens per abstract on that set.

The pre-tokenizer accounts for most of the saving. Ettin uses the GPT-2 pre-tokenizer, which splits terms such as `GLP-1` and `IL-6` at hyphens and digits before BPE runs. BPE then cannot learn those terms as single tokens, whatever corpus it is trained on. With the GPT-2 pre-tokenizer kept, the same retrained 50K table saves only 8%. The first round of this experiment measured that 8% and concluded that domain vocabulary does not help much. The second round traced the small saving to the pre-tokenizer.

No published clinical ModernBERT has tried this. BioClinical ModernBERT and Clinical ModernBERT both ship Ettin's tokenizer unchanged.

Oskar asked the question on 2026-11-06: "could we create an entirely custom domain Ettin model with custom token table ... for life sciences?" This experiment measures only the tokenizer. Whether a model trained on the new table performs well is a separate question, and the follow-up experiment `lifesci-ettin` tests it.
