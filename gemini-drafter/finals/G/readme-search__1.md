## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each query vector in `QUERY_FILE` and prints the nearest matches as JSON Lines.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, with or without a trailing comma.

`remex` writes one JSON object per query:

```
{"query": <line number>, "ids": [...], "scores": [...]}
```

The `query` field is the line number of the query in `QUERY_FILE`. Pass `-k` to set how many results `remex` returns per query; the default is 10.

### Bit width and rescoring

`remex` searches at `--bits 4` by default. On the 100,000-vector EmbeddingGemma test set, 4-bit search reaches 0.93 recall@10 and `--bits 2` reaches 0.84.

If `INDEX` was built with `--keep-float`, `remex` rescores the top 100 candidates for each query against the float vectors. Otherwise it skips rescoring.

### Speed

Searching 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
