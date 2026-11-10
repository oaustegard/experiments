## Searching an index

`remex search` reads query vectors from a file and prints the nearest neighbours for each query as one JSON object per line.

```
remex search INDEX QUERY_FILE
```

### Query file

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is accepted.

```
0.12,-0.48,0.33,0.07
0.91,0.02,-0.15,0.40,
```

### Options

| Flag | Default | Effect |
|------|---------|--------|
| `-k` | `10` | Number of results returned per query |
| `--bits` | `4` | Bits per dimension used for the search |

At 4 bits, recall@10 is 0.93 on the 100,000-vector EmbeddingGemma test set. At 2 bits it is 0.84.

### Rescoring

If the index was built with `--keep-float`, every search rescores its top 100 candidates with the float vectors before it returns results. If the index was built without `--keep-float`, search returns the quantized scores and does no rescoring. Rescoring has no flag of its own.

### Output

Output is JSON Lines, with one object per query:

```json
{"query": 1, "ids": [...], "scores": [...]}
```

`query` is the query's line number in `QUERY_FILE`. `ids` and `scores` hold `k` entries each, in matching order.

### Performance

A search over 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
