## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each vector in `QUERY_FILE` and writes one JSON Lines object per query.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, and the parser accepts lines that end with a trailing comma. Pass `-k` to set how many results each query returns; the default is 10.

`remex` writes its results as JSON Lines, with one object per query:

```json
{"query": 1, "ids": [...], "scores": [...]}
```

The `query` field carries the line number from `QUERY_FILE`.

### Quantization and rescoring

`--bits` defaults to 4. When you search an index that was built with `--keep-float`, `remex` rescores the top 100 candidates for each query against the float vectors; if the index lacks float vectors, `remex` skips rescoring.

On the 100,000-vector EmbeddingGemma test set, 4-bit search reaches 0.93 recall@10 and takes about 3 ms per query on 4 CPU cores; `--bits 2` reaches 0.84 recall@10 on the same set.
