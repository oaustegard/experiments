## Searching an index

`remex search INDEX QUERY_FILE` searches an index for each query vector in `QUERY_FILE` and prints the nearest matches as JSON Lines.

### Input and output

`QUERY_FILE` holds one query vector per line as comma-separated floats, with or without a trailing comma.

`remex` writes one JSON object per query:

```json
{"query": 0, "ids": [...], "scores": [...]}
```

The `query` field is the line number from `QUERY_FILE`. Pass `-k` to set how many results `remex` returns per query; the default is 10.

### Precision, rescoring, and speed

`remex` searches at `--bits 4` by default. On the 100,000-vector EmbeddingGemma test set, 4-bit search achieves 0.93 recall@10, while `--bits 2` achieves 0.84 recall@10.

When `INDEX` was built with `--keep-float`, `remex` rescores the top 100 candidates for each query against the float vectors; otherwise it skips rescoring.

Searching 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
