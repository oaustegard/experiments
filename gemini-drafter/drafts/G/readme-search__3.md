## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each vector in `QUERY_FILE` and prints the nearest neighbours as JSON Lines.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, with or without a trailing comma. Output is JSON Lines with one object per query: `{"query": <line number>, "ids": [...], "scores": [...]}`. Pass `-k` to set the number of results returned per query; the default is `10`.

### Precision, rescoring, and speed

`remex search` runs at `--bits 4` by default. On the 100,000-vector EmbeddingGemma test set, 4-bit search reaches `0.93` recall@10 and takes about `3 ms` per query on 4 CPU cores; passing `--bits 2` reaches `0.84` recall@10.

When you search an index that was built with `--keep-float`, `remex` rescores the top 100 candidates against the stored float vectors; if the index was built without `--keep-float`, `remex` skips rescoring.
