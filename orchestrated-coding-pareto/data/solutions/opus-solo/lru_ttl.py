"""An LRU cache whose entries also expire after a time-to-live."""

from collections import OrderedDict

__all__ = ["TTLCache"]


class TTLCache:
    """LRU cache with per-entry time-to-live expiration.

    Entries are stored in an OrderedDict ordered from least-recently-used to
    most-recently-used. Each entry records the time at which it was inserted or
    last updated; it is considered expired once ``clock() - t >= ttl``.
    """

    def __init__(self, capacity: int, ttl: float, clock):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        # key -> (value, timestamp); ordered LRU (front) .. MRU (back)
        self._data = OrderedDict()

    # -- internal helpers -------------------------------------------------

    def _is_expired(self, timestamp, now) -> bool:
        return now - timestamp >= self.ttl

    def _purge_expired(self, now) -> None:
        expired = [k for k, (_v, t) in self._data.items() if self._is_expired(t, now)]
        for k in expired:
            del self._data[k]

    def _live_count(self, now) -> int:
        return sum(
            1 for _k, (_v, t) in self._data.items() if not self._is_expired(t, now)
        )

    # -- public API -------------------------------------------------------

    def put(self, key, value) -> None:
        now = self.clock()

        if key in self._data:
            # Update refreshes both TTL timestamp and recency, even if expired.
            self._data[key] = (value, now)
            self._data.move_to_end(key)
            return

        if self._live_count(now) >= self.capacity:
            self._purge_expired(now)
            if len(self._data) >= self.capacity:
                # Evict least-recently-used (all remaining are non-expired).
                self._data.popitem(last=False)

        self._data[key] = (value, now)

    def get(self, key, default=None):
        now = self.clock()

        if key not in self._data:
            return default

        value, timestamp = self._data[key]
        if self._is_expired(timestamp, now):
            del self._data[key]
            return default

        self._data.move_to_end(key)
        return value

    def __len__(self) -> int:
        return self._live_count(self.clock())
