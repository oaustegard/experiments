from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity: int, ttl: float, clock):
        self._capacity = capacity
        self._ttl = ttl
        self._clock = clock
        self._data = OrderedDict()  # key -> (value, inserted_at)

    def _is_expired(self, inserted_at) -> bool:
        return self._clock() - inserted_at >= self._ttl

    def _purge_expired(self) -> None:
        expired_keys = [k for k, (_, t) in self._data.items() if self._is_expired(t)]
        for k in expired_keys:
            del self._data[k]

    def put(self, key, value) -> None:
        now = self._clock()
        if key in self._data:
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            return

        self._purge_expired()
        if len(self._data) >= self._capacity:
            self._data.popitem(last=False)

        self._data[key] = (value, now)
        self._data.move_to_end(key)

    def get(self, key, default=None):
        if key not in self._data:
            return default

        value, inserted_at = self._data[key]
        if self._is_expired(inserted_at):
            del self._data[key]
            return default

        self._data.move_to_end(key)
        return value

    def __len__(self) -> int:
        self._purge_expired()
        return len(self._data)
