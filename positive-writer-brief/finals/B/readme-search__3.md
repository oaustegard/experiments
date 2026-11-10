## Searching an index

`remex search` returns the nearest stored vectors for each query vector in a file, written as one JSON object per query.

```sh
remex search INDEX QUERY_FILE
```

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is allowed.

```text
0.12,-0.48,0.33,0.07
0.91,0.02,-0.15,0.40,
```

### Output

The output is JSON Lines, with one object per query. `query` is the line number of the query in `QUERY_FILE`. `ids` and `scores` are parallel arrays.

```json
{"query": 1, "ids": [...], "scores": [...]}
```

### Options

| Flag | Default | Effect |
|------|---------|--------|
| `-k` | 10 | Number of results returned per query |
| `--bits` | 4 | Bit width used for the search |

On the 100,000-vector EmbeddingGemma test set, recall@10 is 0.93 at 4 bits and 0.84 at 2 bits.

### Rescoring

If you built the index with `--keep-float`, every search rescores its top 100 candidates against the float vectors. If the index was built without `--keep-float`, the search skips rescoring.

### Speed

A search of 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
