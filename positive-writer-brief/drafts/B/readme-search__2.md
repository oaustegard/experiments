## Searching an index

`remex search` returns the nearest neighbours for each query vector in a file, as JSON Lines on standard output.

```
remex search INDEX QUERY_FILE
```

### Query file

`QUERY_FILE` holds one query vector per line, written as comma-separated floats. A trailing comma at the end of a line is accepted.

```
0.12,-0.48,0.33,0.91
0.05,0.27,-0.66,0.14,
```

### Output

remex writes one JSON object per query. `query` is the line number of the query in `QUERY_FILE`. `ids` and `scores` list the results in matching order.

```
{"query": 1, "ids": [...], "scores": [...]}
```

### Options

| Option | Default | Effect |
|---|---|---|
| `-k` | 10 | Number of results returned per query |
| `--bits` | 4 | Bits per dimension used for the search |

On the 100,000-vector EmbeddingGemma test set, recall@10 is 0.93 at 4 bits and 0.84 at 2 bits.

### Rescoring

If the index was built with `--keep-float`, every search rescores its top 100 candidates against the float vectors. If it was not, remex returns the results without rescoring. There is no flag that controls this, so build with `--keep-float` if you want rescoring.

### Speed

A search over 100,000 vectors at 4 bits takes about 3 ms per query on 4 CPU cores.
