## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each vector in `QUERY_FILE` and prints the nearest neighbors as JSON Lines.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, with or without a trailing comma. `remex search` writes one JSON object per query:

```json
{"query": 0, "ids": [...], "scores": [...]}
```

The `query` field is the line number from `QUERY_FILE`. Pass `-k` to set the number of results returned per query; the default is 10.

### Quantization and rescoring

`remex search` defaults to `--bits 4`. On the 100,000-vector EmbeddingGemma test set, 4-bit quantization achieves 0.93 recall@10, while `--bits 2` achieves 0.84 recall@10. Searching 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.

When the index was built with `--keep-float`, every search rescores its top 100 candidates against the stored float vectors; when the index omits float vectors, `remex search` skips rescoring.
