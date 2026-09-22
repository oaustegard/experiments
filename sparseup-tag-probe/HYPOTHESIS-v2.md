# Binary Tag Vectors: Second Iteration

A revision of *Automated Domain Specific Document Tagging* (2024-12-05) after
measuring its parts on a real corpus. The corpus is Muninn's memory store:
3,457 short technical notes, each carrying tags a language model assigned when
the note was written, plus 466 explicit citation links between notes. Every
number below is from `oaustegard/experiments/sparseup-tag-probe/`, rounds 1 to 6,
with predictions written before each run.

## The 2024 design

Tags are a sparse binary vector over a tag vocabulary. Extract candidate tags
per document with a parser (entities, noun phrases, important tokens), reduce
the vocabulary to a few hundred dimensions by frequency, embedding
dissimilarity and co-occurrence, order those dimensions so that any prefix is
a coarser view, and compare documents by Hamming or Jaccard distance. Unseen
tags snap onto the selected dimensions by embedding similarity. A later
comment proposed ordering the dimensions by taxonomy level so that truncation
reduces to ancestors.

## Claims that survived measurement

**A binary tag vector retrieves as well as a dense text embedding.** With the
466 citation links as relevance, a binary vector over all 5,827 tags in the
store, about six active per note, finds the cited notes at recall@10 0.667
against 0.648 for gte-small, a 384-dimension text embedding. The confidence
interval on the difference spans zero. On the half of the links whose notes
are furthest apart in time, where a nearest-in-time ranking falls from
0.593 to 0.122, the tag vector holds at 0.534 against 0.566.

**The two views are complementary.** The tag vector and the embedding rank
different neighbours (mean top-10 Jaccard 0.18), the tag vector finds 37% of
what the embedding misses at k = 10, and reciprocal-rank fusion of the two
beats either by 0.06.

**Bits cost almost nothing.** Dropping a learned sparse encoder's weights to
0/1 lost 0.021 average precision on a tag-prediction probe and 0.008 on
precision@5. The binary representation is not where the information goes
missing.

**A sparse vector needs no reduction.** With sparse storage a 5,827-dimension
binary vector costs six integers per document. The pressure to cut it to a
few hundred dimensions came from dense storage and is gone.

## Claims that failed measurement

**The reduce step loses most of the signal.** Selecting a vocabulary of K
dimensions by document frequency, with or without an embedding-dissimilarity
filter, and probing it for the original tags: 512 dimensions carry 0.37 of
the 0.66 average precision that the full word-count matrix carries, and 2,048
carry 0.45. The dissimilarity filter made every size worse, by up to 0.035:
surface variants that look redundant to an embedding are separately
predictive. For retrieval the same holds; 512 selected phrase dimensions score
recall@10 0.223 against 0.667 for the store's own tags.

**Parser extraction is the wrong map step.** Noun chunks and entities from
spaCy, at every vocabulary size, score below plain word unigrams at the same
size, by 0.03 to 0.09, and both score far below the model-assigned tags until
the unigram vocabulary reaches about 20,000 dimensions. Chunking merges the
words that carry a document's identity into rarer compounds.

**Expansion by co-occurrence hurts.** Adding co-occurring tags at PMI weight,
the tag-space analogue of a learned sparse encoder's term expansion, was
within noise of the plain binary vector at the lowest weight tried and cost
0.04 to 0.18 recall@10 at every higher weight, in every sparsity variant. With
six tags per document and co-occurrence learned on a few thousand documents,
the expansion adds neighbours faster than it adds relevant ones.

**A learned sparse encoder's dimensions are not concepts.** SPARSEUP, the
SPLADE-line model that prompted this revision, has dimensions that are
tokenizer pieces. Untrained, a note's tags appear among its top-32 expansion
terms 16 to 24% of the time depending on how a hyphenated tag is counted.
Trained, a linear map from its 50,000-dimension vector to the tags reaches
0.612 average precision, below word and character n-gram TF-IDF at 0.664.

## Where the value turned out to be

The store's tags retrieve because of their rare members. The tags with
document frequency under 20, on their own, reach recall@10 0.742; the tags
with frequency 20 or more, on their own, reach 0.309. Cited pairs share about
three tags, and a fifth of those shared tags occur fewer than five times in the
whole store: a project name, an issue number, a person. Everything the 2024
design did to the vocabulary, frequency ranking, dissimilarity pruning,
reduction to hundreds of dimensions, removes exactly those.

The compression the design wanted, a few meaningful bits per document, is
real, and it comes from the tagging step. Six tags chosen by a model that read
the document do what 20,000 word dimensions do and what 10,000 parser-extracted
phrase dimensions do not.

## The second iteration

**Map step: a model, prompted for proper nouns.** A cheap model,
gemini-3.5-flash-lite at a few cents per thousand documents, re-tagged 300
notes. Shown the 150 most frequent tags as examples of the register, it copied
them 62% of the time and its tags retrieved at 0.331. Told to name the
specific project, repository, tool, paper, person or issue and shown no list,
0.461. The instruction that matters is the opposite of the 2024 selection
criteria: ask for the rare thing.

**Reduce step: canonicalize, do not select.** Keep every tag as a dimension.
Snap each newly written tag onto the nearest existing tag by embedding cosine
above 0.85, else admit it as new. This is the 2024 LSH inference step moved
from query time to write time, and it was worth 0.08 recall@10 on its own.
The one vocabulary cleanup with measured value is merging surface-variant
families (`embedding`/`embeddings`, `correction`/`corrections`; 70 such
families in this store).

**Consistency from context, at write time.** The gap between the cheap
model's tags and the original tags was mostly that the originals were written
with related notes open. Showing the tagger the tags of the five most similar
older notes took it to 0.565. Showing it the tags of the notes the new one
cites took it to 0.629, and on the direction the context cannot leak into,
cited note retrieving its citers, to 0.617 against 0.674 for the originals and
0.626 for the text embedding. This is the two-stage, corpus-then-document
shape the CDE comment pointed at, done with tags instead of embeddings: the
first stage is the neighbourhood's tags, the second is the document.

**Retrieval: fuse, unexpanded.** The binary tag vector by Jaccard or cosine,
reciprocal-rank fused with a text embedding. No PMI expansion.

The pipeline, then:

1. At write time, retrieve the document's cited and nearest older neighbours
   and collect their tags.
2. Ask a small model for four to seven tags, at least half of them the
   specific names in the document, reusing a neighbour's tag exactly when it
   fits.
3. Snap each tag onto the vocabulary at cosine 0.85; admit the rest as new
   dimensions.
4. Store the binary vector beside the embedding; retrieve by fusing the two.

## Untested parts and limits of the evidence

The hierarchy appendix is untested: none of the six rounds measured taxonomy
induction or prefix ordering. What changed is that
the dimensions it would order are the store's own canonical tags, not a
parser's output reduced by frequency. Subsumption from co-occurrence, the
third induction option in the appendix, inherits the caveat that co-occurrence
over a few thousand documents was noisy enough to hurt expansion.

Two limits of the evidence. The relevance set is citations the same writer
made, often after a lexical search, which is part of why n-gram TF-IDF led
every single representation at 0.808. And one corpus, one author, one register
say nothing yet about a bookmark collection with many contributors, which was
the original setting.

## Appendix: the numbers

Retrieval of cited notes, recall@10, 235 queries over 3,457 notes.

| representation | R@10 | note |
|---|---|---|
| store's own tags, binary, 5,827 dims | 0.667 | ties the embedding |
| gte-small, 384 dims | 0.648 | |
| RRF of the two | 0.707 | |
| TF-IDF word + char | 0.808 | relevance favours lexical overlap |
| SPARSEUP document vectors | 0.711 | |
| tag vector + PMI expansion, weight 0.5 | 0.610 | |
| spaCy phrases, 512 selected dims | 0.223 | the 2024 map + reduce |
| word unigrams, all 19,563 dims | 0.701 | |
| nearest in time | 0.593 | 0.122 on the distant half |

Cheap tagger on 300 notes, recall@10 of cited notes, same test.

| tags written by | forward | reverse (leak-free) |
|---|---|---|
| original writer, with the session open | 0.697 | 0.674 |
| flash-lite, frequent-tag hint prompt | 0.331 | |
| flash-lite, proper-noun prompt | 0.461 | |
| flash-lite, + five older neighbours' tags | 0.565 | 0.560 |
| flash-lite, + cited notes' tags | 0.629 | 0.617 |
| gte-small on the same queries | 0.719 | 0.626 |
