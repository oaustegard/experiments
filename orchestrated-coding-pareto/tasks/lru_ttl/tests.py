from solution import TTLCache


class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t


def test_basic_put_get():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    assert cache.get("a") == 1

def test_missing_returns_default():
    cache = TTLCache(2, 10.0, Clock())
    assert cache.get("nope") is None
    assert cache.get("nope", 42) == 42

def test_expiry_at_exactly_ttl():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 10.0
    assert cache.get("a") is None

def test_not_expired_just_before_ttl():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 9.999
    assert cache.get("a") == 1

def test_lru_eviction_order():
    c = Clock()
    cache = TTLCache(2, 100.0, c)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts a
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_get_refreshes_recency():
    c = Clock()
    cache = TTLCache(2, 100.0, c)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a is now MRU
    cache.put("c", 3)           # evicts b, not a
    assert cache.get("b") is None
    assert cache.get("a") == 1

def test_put_existing_refreshes_ttl():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 8.0
    cache.put("a", 2)
    c.t = 15.0  # 7 after refresh, would be 15 after original insert
    assert cache.get("a") == 2

def test_expired_dropped_before_eviction():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 5.0
    cache.put("b", 2)
    c.t = 12.0  # a expired, b alive
    cache.put("c", 3)  # a dropped as expired; b must survive
    assert cache.get("b") == 2
    assert cache.get("c") == 3

def test_len_counts_only_live():
    c = Clock()
    cache = TTLCache(5, 10.0, c)
    cache.put("a", 1)
    c.t = 5.0
    cache.put("b", 2)
    assert len(cache) == 2
    c.t = 11.0
    assert len(cache) == 1
    c.t = 20.0
    assert len(cache) == 0

def test_get_expired_removes_entry():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 11.0
    assert cache.get("a") is None
    c.t = 0.0  # even if time went backwards, entry is gone
    assert cache.get("a") is None

def test_update_expired_key_revives():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 11.0
    cache.put("a", 2)
    assert cache.get("a") == 2

def test_value_none_is_storable():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", None)
    assert cache.get("a", "sentinel") is None
