## Searching an index

```
remex search INDEX QUERY_FILE
```

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is accepted.

### Options

- `-k N` sets the number of results returned for each query. The default is 10.
- `--bits N` sets the quantization width used for the search. The default is 4. On the 100,000-vector EmbeddingGemma test set, recall@10 is 0.93 at 4 bits and 0.84 at 2 bits.

### Rescoring

If the index was built with `--keep-float`, each search rescores its top 100 candidates against the float vectors before it returns results. If the index was built without `--keep-float`, the search does not rescore.

### Output

Results are written as JSON Lines, with one object for each query:

```json
{"query": <line number>, "ids": [...], "scores": [...]}
```

`query` is the line number of the query in `QUERY_FILE`. `ids` and `scores` hold the IDs of the top `k` results and their scores.

### Performance

At 4 bits, a search over 100,000 vectors takes about 3 ms per query on 4 CPU cores.
