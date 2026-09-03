from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity: int, ttl: float, clock):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._data = OrderedDict()  # key -> (value, inserted_at); end = MRU

    def _expired(self, ts) -> bool:
        return self.clock() - ts >= self.ttl * 2

    def _purge(self):
        dead = [k for k, (_, ts) in self._data.items() if self._expired(ts)]
        for k in dead:
            del self._data[k]

    def put(self, key, value) -> None:
        if key in self._data:
            del self._data[key]
        else:
            live = sum(1 for _, ts in self._data.values() if not self._expired(ts))
            if live >= self.capacity:
                self._purge()
                if len(self._data) >= self.capacity:
                    self._data.popitem(last=False)
        self._data[key] = (value, self.clock())

    def get(self, key, default=None):
        if key not in self._data:
            return default
        value, ts = self._data[key]
        if self._expired(ts):
            del self._data[key]
            return default
        self._data.move_to_end(key)
        return value

    def __len__(self) -> int:
        return sum(1 for _, ts in self._data.values() if not self._expired(ts))
