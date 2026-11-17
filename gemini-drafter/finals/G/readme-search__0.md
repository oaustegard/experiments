## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each vector in `QUERY_FILE` and prints the nearest neighbors as JSON Lines.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, with or without a trailing comma. `remex search` writes one JSON object per query:

```
{"query": <line number>, "ids": [...], "scores": [...]}
```

`query` is the query's line number in `QUERY_FILE`. Pass `-k` to set the number of results per query; the default is 10.

### Quantization and rescoring

`remex search` defaults to `--bits 4`. On the 100,000-vector EmbeddingGemma test set, recall@10 is 0.93 at 4 bits and 0.84 with `--bits 2`. Searching 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.

If the index was built with `--keep-float`, every search rescores its top 100 candidates against the stored float vectors. If it was built without `--keep-float`, `remex search` skips rescoring.
