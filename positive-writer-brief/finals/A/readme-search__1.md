## Searching an index

```
remex search INDEX QUERY_FILE
```

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is accepted.

`-k` sets how many results each query returns. The default is 10.

`--bits` sets the quantization width used for the search. The default is 4. On the 100,000-vector EmbeddingGemma test set, recall@10 is 0.93 at 4 bits and 0.84 at 2 bits. A search over 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.

Rescoring depends only on how the index was built. If it was built with `--keep-float`, each search rescores its top 100 candidates against the float vectors before it returns results. If it was built without `--keep-float`, the search skips rescoring.

Output is JSON Lines, with one object per query:

```json
{"query": 1, "ids": [...], "scores": [...]}
```

`query` is the query's line number in `QUERY_FILE`. `scores[i]` is the score for `ids[i]`.
