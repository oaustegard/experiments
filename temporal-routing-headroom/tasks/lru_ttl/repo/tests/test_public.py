from solution import TTLCache
class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t


def test_expiry_at_exactly_ttl():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 10.0
    assert cache.get("a") is None


def test_get_expired_removes_entry():
    c = Clock()
    cache = TTLCache(2, 10.0, c)
    cache.put("a", 1)
    c.t = 11.0
    assert cache.get("a") is None
    c.t = 0.0  # even if time went backwards, entry is gone
    assert cache.get("a") is None
