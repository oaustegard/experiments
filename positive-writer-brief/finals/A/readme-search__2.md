## Searching an index

```
remex search INDEX QUERY_FILE
```

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is accepted.

### Options

- `-k` sets how many results to return for each query. The default is 10.
- `--bits` sets the quantization width used for the search. The default is 4. On the 100,000-vector EmbeddingGemma test set, 4-bit search has a recall@10 of 0.93, and 2-bit search has a recall@10 of 0.84.

### Rescoring

If the index was built with `--keep-float`, every search rescores its top 100 candidates against the stored float vectors before it returns results. If the index was built without `--keep-float`, the search skips rescoring.

### Output

Results are written as JSON Lines, with one object for each query:

```json
{"query": <line number>, "ids": [...], "scores": [...]}
```

`query` is the line number of the query in `QUERY_FILE`. `ids` holds the IDs of the matched vectors, and `scores` holds their scores.

### Speed

A search over 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
