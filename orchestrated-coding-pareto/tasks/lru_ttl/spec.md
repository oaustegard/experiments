# Task: lru_ttl

Write a Python module defining exactly one public class:

```python
class TTLCache:
    def __init__(self, capacity: int, ttl: float, clock):
        ...
    def put(self, key, value) -> None: ...
    def get(self, key, default=None): ...
    def __len__(self) -> int: ...
```

An LRU cache where entries also expire after a time-to-live.

Rules:
- `clock` is a zero-argument callable returning the current time as a float (injected for
  testability; never use `time.time()` yourself).
- An entry inserted (or updated) at time `t` is **expired** once `clock() - t >= ttl`
  (note: `>=`, so an entry is already expired exactly `ttl` after insertion).
- `put(key, value)`: inserts or updates. Updating an existing (even expired) key refreshes
  both its TTL timestamp and its recency. If inserting a **new** key when the number of
  non-expired entries equals `capacity`, first drop all expired entries, then evict the
  least-recently-used non-expired entry.
- `get(key, default=None)`: returns the value if the key exists and is not expired, and
  marks it most-recently-used. If the key is missing or expired, returns `default`; an
  expired entry hit by `get` is removed.
- `len(cache)` returns the number of **non-expired** entries currently stored.
- Recency: both `put` and a successful `get` make the key most-recently-used.
- `capacity` is always >= 1.

No I/O. Standard library only. Only `TTLCache` is tested.
