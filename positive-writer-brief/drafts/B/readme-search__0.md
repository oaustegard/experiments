## Searching an index

`remex search` reads query vectors from a file and prints the nearest neighbours for each query, one JSON object per line.

```
remex search INDEX QUERY_FILE
```

### Query file

Put one query vector on each line of `QUERY_FILE`, written as comma-separated floats:

```
0.12,-0.48,0.33,0.07
-0.91,0.25,0.10,0.66
```

A trailing comma at the end of a line is allowed.

### Options

| Flag | Default | Effect |
|------|---------|--------|
| `-k` | 10 | Number of results returned per query |
| `--bits` | 4 | Bits per dimension used for the search |

At 4 bits, recall@10 is 0.93 on the 100,000-vector EmbeddingGemma test set. At 2 bits it is 0.84. On 4 CPU cores, a 4-bit search over 100,000 vectors takes about 3 ms per query.

### Rescoring

If the index was built with `--keep-float`, every search rescores its top 100 candidates with the float vectors. If it was built without `--keep-float`, the search skips rescoring. No flag controls this, so the way you built the index determines whether rescoring happens.

### Output

The output is JSON Lines, with one object per query:

```json
{"query": 1, "ids": [...], "scores": [...]}
```

`query` is the query's line number in `QUERY_FILE`. `ids` and `scores` hold one entry for each of the `k` results.
