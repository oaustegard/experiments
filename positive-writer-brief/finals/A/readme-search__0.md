## Searching an index

```
remex search INDEX QUERY_FILE
```

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is allowed.

### Options

- `-k` sets how many results are returned for each query. The default is 10.
- `--bits` sets the quantization width used for the search. The default is 4. On the 100,000-vector EmbeddingGemma test set, 4 bits gives a recall@10 of 0.93 and 2 bits gives 0.84.

### Rescoring

If the index was built with `--keep-float`, every search rescores its top 100 candidates against the float vectors. If it was built without `--keep-float`, remex skips rescoring.

### Output

Results are written as JSON Lines, with one object per query:

```json
{"query": 1, "ids": [...], "scores": [...]}
```

`query` is the line number of the query in `QUERY_FILE`. The `ids` and `scores` arrays hold the results for that query.

### Speed

At 4 bits, a search over 100,000 vectors takes about 3 ms per query on 4 CPU cores.
