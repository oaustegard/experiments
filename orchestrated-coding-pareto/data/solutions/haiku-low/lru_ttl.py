from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity: int, ttl: float, clock):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.cache = {}  # key -> (value, timestamp)
        self.order = OrderedDict()  # track insertion/access order for LRU

    def _is_expired(self, key):
        """Check if a key's entry is expired."""
        if key not in self.cache:
            return False
        value, timestamp = self.cache[key]
        return self.clock() - timestamp >= self.ttl

    def put(self, key, value) -> None:
        is_new = key not in self.cache

        if is_new:
            # Count non-expired entries
            non_expired_count = sum(1 for k in self.cache if not self._is_expired(k))

            if non_expired_count >= self.capacity:
                # Remove all expired entries
                expired_keys = [k for k in self.cache if self._is_expired(k)]
                for k in expired_keys:
                    del self.cache[k]
                    del self.order[k]

                # Recount non-expired entries
                non_expired_count = sum(1 for k in self.cache if not self._is_expired(k))

                # Evict LRU non-expired entry if still at capacity
                if non_expired_count >= self.capacity:
                    for k in self.order:
                        if not self._is_expired(k):
                            del self.cache[k]
                            del self.order[k]
                            break

        # Insert or update the entry
        self.cache[key] = (value, self.clock())

        # Update recency (move to end of order)
        if key in self.order:
            del self.order[key]
        self.order[key] = True

    def get(self, key, default=None):
        if key not in self.cache:
            return default

        if self._is_expired(key):
            # Remove the expired entry
            del self.cache[key]
            del self.order[key]
            return default

        # Mark as most-recently-used
        del self.order[key]
        self.order[key] = True

        value, timestamp = self.cache[key]
        return value

    def __len__(self) -> int:
        # Return count of non-expired entries
        return sum(1 for k in self.cache if not self._is_expired(k))
